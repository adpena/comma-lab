# ddm_jo4_certified_retention_reseal — cure the two r6 scale blockers → seal r7 READY_TO_FIRE_UNDER_STANDING_GO

## MISSION
jo3 (memo .omx/research/ddm_jo3_entrypoint_and_final_reseal_20260821.md) BUILT the real
local CPU entrypoint (experiments/ddm_jo3_joint_objective_entrypoint.py, sha
92d2a2ab2a927d15dcdc1b97642edfdd4ceaf414113a3ad342b3423760c1f4a6) and cleared four r5
blockers; seal r6 (.omx/research/ddm_jo3_entrypoint_and_final_reseal_20260821/seal_r6/,
compiled sha 3af9848e…) blocks on exactly two REAL preflight facts. Cure both, re-run
the real-config preflight, reseal r7 READY_TO_FIRE_UNDER_STANDING_GO. MAIN fires the
20.9–35.4 h governed local run; you launch NOTHING heavy.

## MAIN POLICY ADJUDICATION (binding for this arm — the storage blocker's cure)
Blocker RETAINED_FRESH_SCHUR_STORAGE_BLOCKED (2,907,449,989,536 B ≥ 4.8× the 603 GB
free tier) arises from retaining FULL uncompressed camera bytes for EVERY explored
candidate (≥103,972 candidates/stage × 9,156,024 B). MAIN adjudicates per the
CLAUDE.md "Local Disk, SSD Spill" certify-or-block rule (the P0 ALWAYS-KEEP-THE-PAYLOAD
section's own storage-ROUTING clause): deterministically-rebuildable non-winner
candidates satisfy the P0 via CERTIFIED REBUILD RECORDS, never full-byte retention.
Implement a two-tier retention mode in the entrypoint:
- FULL BYTES (unchanged): every stage WINNER object per pair (carrier state, camera
  payloads, PoseNet IO, coder-race artifacts), every stage-exit checkpoint, admission
  receipts, determinism repeats.
- CERTIFIED-REBUILDABLE (non-winner explored candidates): at explore time, compute and
  persist per candidate {sha256 of the materialized camera payload, byte count, and the
  exact regeneration tuple (entrypoint sha, workload identity, base archive sha
  4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841, stage id, pair id,
  candidate coordinate delta)} — then release the buffer. The receiver's determinism is
  RECEIPTED (r8 dual-decode 0.raw byte-identical, sha 6bf8acf8d4412e43f8ddf810bcf63feb…,
  MATERIALIZER_RESULT.json deterministic_repeat.byte_identical=true), so the record
  preserves exact reproducibility — the machine-readable proof the rule demands.
This is NOT the measure-and-discard fake (the detector signature is "only persisted
artifact is scalars WITH NO adjacent deterministic-rebuild certification"); the cert
rows + winner bytes + hash-at-explore-time satisfy both the letter and the detector.
Projected retention under this mode: ~5.5 GB winners/stage + ~21 MB certs/stage + fields
and checkpoints ≪ 603 GB. RE-DERIVE this bound in the preflight; do not cite mine.

## WORK ORDER (strict order)
1. ENDPOINT-SAFE DERIVATIVES: at the 9 measured int12 endpoint coordinates
   ((63,10),(67,10),(150,0),(150,7),(162,6),(214,8),(252,11),(450,9),(543,4)) replace
   central differences with one-sided differences at matched step (document the
   truncation-order change in code; assert in-domain for BOTH probe points at every
   coordinate). jo3's "endpoint-safe lower-bound formulation" already scoped this —
   implement it in the pinned solve path, never a fork.
2. CERTIFIED RETENTION MODE per the adjudication above — implemented IN the entrypoint's
   retention layer, fail-closed: a candidate whose cert row cannot be written refuses
   the candidate (never silently skips retention).
3. RE-RUN the real-config preflight (same receipt schema as r6's MEMORY_PREFLIGHT.json,
   09e5affa…) — memory geometry re-verified, NEW storage bound computed from the
   two-tier mode, endpoint census re-run (should report 0 blocked coordinates).
4. RESEAL r7 via the landed resealer (pin TRIPLE from the working tree) →
   READINESS READY_TO_FIRE_UNDER_STANDING_GO, FIRE_ORDER with complete argv at every
   ordinal (r6's ordinal-3 argv:null must be filled). Commit via serializer
   (post-edit --expected-content-sha256); .py = 2 genuine review passes.

## OPTIMAL FORM
Family reference form + receipt: the jo1 solve family's reference is jo3's landed
entrypoint (three real stages, through-R frozen SegNet/PoseNet, exact float16 residual
receiver, pinned solve_fresh_compensation, real coder race, decoded identity, exact
stage admission — receipt seal_r6 READINESS sha 79b571c9…, one-pair probe 1.3895 s/step,
grad norm 0.0012942249, peak RSS 2.86 GB, streamed n600 projection 5.36 GB PASS) + the
qs5 in-compile compensation receipt
(.omx/research/ddm_qs5_verdict_and_no_toy_enforcement_20260813.md, d_pose below base,
repeat identical). The r7 seal must invoke THAT form at full scale.
Provenance pin: experiments/ddm_jo3_joint_objective_entrypoint.py=92d2a2ab2a927d15dcdc1b97642edfdd4ceaf414113a3ad342b3423760c1f4a6
(also pinned: receiver-close f391b719… + residual runtime 455b1b2d… per the r6 table).
SCOPE reductions (legal): one-pair/3-pair probe re-runs for the preflight. MECHANISM
reductions: NONE — the pinned solver mechanism is untouched except the two named cures;
a weakened-solve or skip-retention variant is the exact fake jo2/jo3 refused.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends, accounted)
- r6's own honest blockers (memo .omx/research/ddm_jo3_entrypoint_and_final_reseal_20260821.md):
  a READY seal at r6 "would have been a fake" — the cures here must be REAL (endpoint
  asserts executed, storage bound re-derived), not renamed constants.
- qs4 REFUSED +2.437870e-4 stale-compensation (memo
  .omx/research/ddm_qs5_verdict_and_no_toy_enforcement_20260813.md): four-way object
  binding stays intact through both cures.
- jo2 bounded-search false-MISSING (memo .omx/research/ddm_jo2_solve_reseal_20260821.md):
  any "missing/insufficient" claim must name its exact measured scope.
- pk4 linear-overlay FORMULATION ceiling (memo
  .omx/research/ddm_pk4_optimal_form_frame0_pose_verdict_20260813.md): do not regress
  any stage to linear per-pair overlays.

## CONTEXT ANCHORS (memo-associated)
- Campaign sub-0.12 #1182 (memo .omx/research/ddm_r012_rate_representation_20260821.md);
  pointer fx5_e1 S 0.14823186109359 @ 180,386 B [contest-CUDA T4 n600].
- Payload custody (triple-bound by jo3, VERIFIED — do not re-download):
  /Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/payloads_r8/
  fx5_e1_argmax_n600.npy (e89e1ac0…) + fx5_e1_first6_n600.npy (71f7d263…).
- Preflight receipt: experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo3_entrypoint_final_reseal_20260821_r6_final/memory_preflight/MEMORY_PREFLIGHT.json.

## CONTRACT
Serializer commits only; never REVIEW_GATE_OVERRIDE=1 on .py; upstream/ READ-ONLY; end
with a final message stating r7 status + blocker delta + the exact FIRE_ORDER path +
the re-derived storage bound.
