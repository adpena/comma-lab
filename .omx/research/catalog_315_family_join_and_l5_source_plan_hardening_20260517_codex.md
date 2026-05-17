# Catalog #315 Family Join + L5 Source-Plan Custody Hardening

Date: 2026-05-17
Author: Codex
Scope: production hardening; no provider dispatch; no score claim

## Why This Exists

The substrate-design meta-assumption review surfaced a concrete control-plane
failure: a substrate can be implemented at L1 while its latest council anchor is
still `PROCEED_WITH_REVISIONS`, and the paid-dispatch gate must not treat
implementation completeness as optimal-form evidence.

Catalog #315 already encoded that discipline, but the live state showed two
opposite risks:

- false negatives: council posterior IDs often use surface names such as
  `nscs01_nullspace_split_renderer_v1_head0_capacity_surface` while registry
  lanes use IDs such as `lane_nscs01_nullspace_split_renderer_20260515`;
- false positives: broad family tokens such as `z3` and `time_traveler` join
  unrelated lanes to revised council anchors.

## Changes Landed

1. Catalog #315 now has a separator-aware, narrow substrate-family fallback
   join after exact ID lookup fails.
2. Broad ambiguous tokens were kept out of the fuzzy join; concrete family
   tokens such as `z3_g1`, `z3_balle`, `nscs01`, `nscs03`, and `z6` are used
   instead.
3. Council/meta lanes are excluded from the dispatch-scope classifier.
4. Catalog #315 v2 treats any non-positive council verdict as not authorized
   for paid dispatch. `DEFER`, `REFUSE`, `ESCALATE`, and unknown verdicts are
   blockers until a positive `PROCEED` or `PROCEED_UNCONDITIONAL` supersedes
   them.
5. The gate does not retroactively require council rows for every historical
   substrate lane. After exact-ID and family-token join both fail, the lane is
   left to sister evidence gates; Catalog #315 is the council-verdict-binding
   surface, not a broad migration gate.
6. Live registry dispositions were made explicit:
   - `lane_nscs01_nullspace_split_renderer_20260515` is `research_only=true`
     until head0-capacity revisions are applied and a follow-on positive
     PROCEED anchor lands.
   - `lane_phase_1b_z6_lift_20260516` is `research_only=true` until recipe
     blockers clear and a follow-on positive PROCEED anchor lands.
7. L5 v2 TT5L side-info Lightning dry-run verification now rejects absolute
   source-plan paths and parent-directory traversal before SHA verification.
   This prevents a hand-edited bundle from pointing at out-of-repo private state
   with a matching hash and weakening source-of-truth custody.

## Regression Coverage

Focused tests cover:

- separator-aware family-token extraction;
- NSCS surface-ID to lane-ID recovery;
- no false match from `z3_balle` to `z3_g1`;
- no false match from generic `time_traveler_l5` lanes to Z6;
- no council/meta lane inclusion;
- non-positive verdicts blocking paid dispatch;
- live Catalog #315 count staying zero after explicit opt-outs;
- L5 source-plan absolute-path rejection;
- L5 source-plan parent-traversal rejection.

## Evidence Grade

This is a guard/custody hardening artifact, not a score movement claim.
It improves dispatch correctness for L5 staircase and substrate-lattice work by
making "implemented" and "authorized/optimal enough for paid dispatch" distinct
states in the control plane.
