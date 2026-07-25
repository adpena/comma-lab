# DDM j8f counted DM4-to-J5 application operator

## Authority and execution surface

- Delegated authority: `/Users/adpena/Projects/pact/.omx/tmp/codex_runs/ddm_j8f_counted_application_operator_20260724T181414Z.wrapped.prompt.txt`
  (`sha256=8b86a8f6e2016b10ab70053e5d5ee5444f9e3bfb5001a169bbd5a6cbe8803d9d`,
  `bytes=6713`).
- Lane: `lane_ddm_j8f_counted_application_20260724`.
- This is an already-arbitraged Fable-to-Codex implementation lane. It stays local,
  uses four Torch threads, spends `$0`, and does not dispatch or start a campaign.
- The worktree branch is an isolated review branch. MAIN retains FIRE authority and
  must review/merge every commit.

## Problem

J8e exposes six hash-bound scorer-recursive DM4 descriptors, but the adapter is
deliberately descriptor-only. The missing object is a typed operator that maps each
descriptor into the existing 368-coordinate J5 receiver grammar, materializes exact
archive bytes, verifies receiver parse-back, and prices the result with the exact
joint score.

## Build

1. Add a typed `ddm_dm4_j5_counted_application.v1` operator.
   - Reconstruct the descriptor's exact scorer-plane target/secant on its stored
     stride-2 stem support.
   - Lift that descriptor through the exact factor-2 preimage used by DM4.
   - Enumerate only J5 coordinates that are receiver-effective at the descriptor's
     pair plus the shared counted template coordinates.
   - Measure each `+1/-1` coordinate by compiling and parsing a real J5 archive. No
     uncounted pixel write is admitted.
   - Select an integer coordinate using the exact `{pair,bucket}` MS4d Hessian and
     adjoint: `p_N=-H^+g`. Candidate magnitudes are exact one-quantum receiver
     secants; there is no global learning rate.
   - Emit descriptor, coordinate, realized uint8, archive-byte, projection, and
     curvature custody in every receipt.

2. Make trust-region behavior fail closed.
   - V16/v17 explicitly do not provide a transferable J5-coordinate validity curve.
     Therefore this smoke has no invented shrink/grow factors.
   - The conservative bound is the smallest receiver lattice: one quantum per
     coordinate, with a coordinate used at most once from the Step-4 anchor.
   - Receipt blocker/gap:
     `J5_BUCKET_VALIDITY_RADIUS_CURVE_ABSENT_NO_SHRINK_GROW_TRANSFER`.

3. Add the operator-directed in-loop #580 hygiene pass.
   - Project each selected receiver effect through the existing fp64
     `P_range(A)` implementation after selection and before parameter commit.
   - Map the projected effect back to the nearest unused exact J5 integer secant,
     using stable coordinate order as the gauge representative.
   - Record rejected null/gauge energy per step.
   - Preserve both the raw and projected aggregate candidates for exact A/B replay;
     the projected arm is admissible only when realized joint delta is
     unchanged-or-better.

4. Wire the operator to the J8e proposal source without mutating the sealed J8e
   ticket.
   - Load J8e through `DirectDescriptionJointDescentTypedConfigV1`.
   - Obtain the six descriptors only through `dm4_j5_proposal_source`.
   - Bind the J8f application config to the J8e ticket, DM4 receipt/config, MS4d
     direct metric, Step-4 checkpoint/verdict, validity law/receipt, NCDE observer,
     and #580 projector by exact bytes and SHA-256.

5. Run one bounded, resumable Step-4 smoke with two exact final arms.
   - No J5-specific measured NCDE re-entry-time receipt exists. The conservative
     horizon is the canonical #344 trailing window of 12 verdict points (minimum
     identifiability is 8), named as
     `J5_NCDE_REENTRY_TIME_CUSTODY_ABSENT_USING_CANONICAL_WINDOW_12`.
   - Cycle the six descriptors twice, producing 12 atomically preserved local
     application checkpoints. Every selected coordinate remains within its
     one-quantum validity bound.
   - Compile one raw aggregate and one range/gauge-projected aggregate.
   - Run exact chunked n600 frozen-scorer replay on both aggregates from the same
     Step-4 reference. This is one A/B re-smoke, not a campaign.
   - `projected_delta_S < 0` and `projected_delta_S <= raw_delta_S` yields a resealed
     `READY_TO_FIRE_DDM_EVENT_CONTINUATION` ticket for MAIN review. Otherwise emit a
     SHA-bound decomposition and a formulation/instance-scoped blocker.

## Resumability, storage, and cleanup

- Output bulk goes to `/Volumes/VertigoDataTier/pact`, falling back to
  `/Volumes/APDataStore/pact`; local bulk is refused.
- Preflight validates real config/source hashes, four-thread scorer custody, RAM,
  and storage before materialization.
- Each pair's exact sparse J5 secant inventory is an atomic, SHA-bound SSD
  checkpoint keyed by Step-4 archive/theta, parameter names, pair, and operator
  source. A restart can lose at most the currently compiling pair; preflight and
  application reuse the same preserved bytes.
- Every horizon step writes a distinct atomic checkpoint containing the selected
  raw/projected coordinates, cumulative theta, archive hashes, and sufficient
  receipt state to resume at the next descriptor.
- Dense candidate cameras remain in-process scratch and are released after each
  pair. Sparse exact effects, final archives, and all checkpoints are preserved.

## Verification

- Focused unit and integration tests, three clean passes.
- Disabled adapter remains byte-identical.
- Every coordinate mutation changes counted archive bytes and survives parse-back.
- MS4d pair/bucket join, Newton pseudoinverse, deterministic tie-breaking,
  one-quantum/no-reuse trust bound, #580 rejected-energy telemetry, resume identity,
  and raw/projected exact comparison are covered.
- Final findings memo:
  `.omx/research/codex_findings_ddm_j8f_<utc>_codex.md`.
- Pointer remains `0.1910828242 [contest-CPU] UNMOVED`; all local scorer evidence is
  `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`.
