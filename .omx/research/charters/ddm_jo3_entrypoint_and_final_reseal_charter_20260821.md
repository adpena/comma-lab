# ddm_jo3_entrypoint_and_final_reseal — drive jo1 seal r5 → r6 READY_TO_FIRE_UNDER_STANDING_GO

## MISSION
Close the LAST build gap in the jo1 joint-objective solve chain and reseal to
READY_TO_FIRE_UNDER_STANDING_GO. jo2 (commit 298b86c543, memo
.omx/research/ddm_jo2_solve_reseal_20260821.md) landed the real receiver-close +
residual runtime with tests; seal r5
(.omx/research/ddm_jo2_solve_reseal_20260821/seal_r5/, compiled sha
35487b801d35916de3f9ad252fbae81801d71993815a3914ea2fe3da1b29620f) is honest-BLOCKED
with exactly four blockers. Two of them are ALREADY DISSOLVED by facts below. You own
the remaining two. MAIN fires the governed launch; you launch NOTHING heavy.

## FACT 1 — the two "MISSING" base payloads EXIST in verified local custody
jo2's bounded search missed them; MAIN re-verified 2026-08-21 (shas match the COMPLETE
materializer receipt at
/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/harvest_r8/MATERIALIZER_RESULT.json).
Use these EXACT absolute paths — do NOT search, do NOT re-download from Modal:
- rc2_base_argmax_field =
  /Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/payloads_r8/fx5_e1_argmax_n600.npy
  117,964,928 B, sha256 e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34,
  uint8 (600,384,512). Valid for rc2: fx5_e1 decode is BYTE-IDENTICAL to rc2 (0.raw sha
  6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883, dual-decode proven).
- fx5_base_pose6 =
  /Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/payloads_r8/fx5_e1_first6_n600.npy
  14,528 B, sha256 71f7d2639eb624f4d0eb89e40ac5956a74b1f72951dc7f07424468769af8350f,
  float32 (600,6). This is the fx5 BODY output (compensation-preservation target), NOT
  the DALI source target — jo2's two-slot distinction is binding; never substitute.
Re-verify both shas yourself (shasum -a 256), then fill the two null input slots via the
resealer (never hand-edit compiled_config.json). Ignore ._* AppleDouble sidecars (#1122).

## WORK ORDER (strict order)
1. RE-POINT: fill rc2_base_argmax_field + fx5_base_pose6 from FACT 1 through the jo2
   resealer path (pin TRIPLE law: sha256 + source_object_sha256 + bytes updated together
   from the working tree — partial updates cost PIN_DRIFT rounds).
2. ENTRYPOINT (the real build): implement the trainer entrypoint that INVOKES the landed
   primitives (experiments/ddm_jo2_receiver_close.py f391b719… +
   experiments/ddm_jo2_residual_runtime.py 455b1b2d…) for the three REQUIRED_STAGE_IDS
   (target_birth, joint_balance, collateral_finish) per
   experiments/ddm_jo1_joint_objective_design.py (r5 pin 9f41216e…). ADJUDICATE
   local-vs-Modal per the compute-split law (Modal only-if-impossible-locally + short):
   scratch is already re-rooted to local APFS
   experiments/.scratch/ddm_jo2_joint_objective_solve (603 GB free, measured r5) and the
   solve is CPU-torch PoseNet forwards — derive the local wall-clock estimate first; if
   local is feasible, the entrypoint is a LOCAL governed-launcher target and the Modal
   leg is dropped from the seal. Keep jo2's refuse-instead-of-pretend discipline: no
   stage may pass without its primitives actually running (NO-FAKE #1). Resumable +
   per-stage checkpoints P0.
3. MEMORY PREFLIGHT: produce the receipt at the REAL config (peak RSS projection at
   n600, chunked verdict per the #205 law) → clears blocker 4.
4. RESEAL r6 via the landed resealer → READINESS status READY_TO_FIRE_UNDER_STANDING_GO
   with zero blockers, FIRE_ORDER naming the exact launch command for MAIN. Commit via
   serializer (post-edit --expected-content-sha256); .py = 2 genuine review passes.

## OPTIMAL FORM
Family reference form + receipt: the jo1 joint-objective solve family's reference is the
jo2-landed receiver-close chain (fresh per-pair central-difference Schur/GN + quantized
neighborhood search + exact coordinate descent, four-way object binding), receipted in
seal r5 READINESS.json sha f25c2f4a9cf6ca58bbe674eada53b5f17567b9d5be5deea114ef535a7d0dd75e
and the jo2 receiver control runs
(/Volumes/APDataStore/pact/ddm_jo1_joint_objective_design/receiver_control_r{1,2}/); the
qs5 compensation mechanism receipt is
.omx/research/ddm_qs5_verdict_and_no_toy_enforcement_20260813.md (d_pose below base,
repeat identical). The entrypoint must invoke THAT form at full scale — n600, all three
stages, real coders.
Provenance pin: experiments/ddm_jo2_receiver_close.py=f391b71963f6cd69611edac10df44408a49aa824942c31c3305d7971386edf5a
(the landed receiver-close this entrypoint must invoke; r5 table also pins
ddm_jo1_joint_objective_design.py=9f41216e3ed0df33586dba101bda5ca7f408c4dd4b6670bb0b220536fa6e9f1d).
SCOPE reductions (legal): smoke the entrypoint on ≤3 pairs before the full-config
preflight. MECHANISM reductions: NONE — the entrypoint must run the REAL receiver-close
and residual runtime, never stand-in stubs; a stub-invoking entrypoint is the exact fake
jo2 refused to write.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends, accounted)
- qs4 REFUSED +2.437870e-4 (stale Schur compensation carried cross-object — memo
  .omx/research/ddm_qs5_verdict_and_no_toy_enforcement_20260813.md): jo2's four-way
  object binding makes this unrepresentable; the entrypoint MUST preserve that binding.
- qs5 R2 REFUSED +2.519822e-6 (same memo): in-compile compensation PROVEN (pose below
  base) — the mechanism your stages inherit; the economics must be re-won jointly.
- pk4 FORMULATION ceiling (memo
  .omx/research/ddm_pk4_optimal_form_frame0_pose_verdict_20260813.md): linear frame-0
  overlays dead — jo1's joint/nonlinear solve is the surviving route; do not regress to
  linear per-pair overlays inside any stage.
- jo2's own bounded-search false-MISSING (memo
  .omx/research/ddm_jo2_solve_reseal_20260821.md §Payload and storage custody): the
  negative-existence claim class — this charter's cure is verbatim absolute paths; if
  ANY file you need is "missing", report the exact search scope, never a bare MISSING.

## CONTEXT ANCHORS (memo-associated)
- Campaign: sub-0.12 #1182 (memo .omx/research/ddm_r012_rate_representation_20260821.md);
  pointer fx5_e1 S 0.14823186109359 @ 180,386 B [contest-CUDA T4 n600].
- Materializer receipt (payload provenance):
  /Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/harvest_r8/MATERIALIZER_RESULT.json
  (schema ddm_jo1_payload_materializer_result.v1; deterministic repeat byte-identical).
- Seal custody: .omx/research/ddm_jo2_solve_reseal_20260821/seal_r5/{READINESS,FIRE_ORDER,author_config,compiled_config}.json.

## CONTRACT
Serializer commits only; never REVIEW_GATE_OVERRIDE=1 on .py; keep the payloads (P0 DEF
CON 1000 — every materialized artifact persisted with sha+bytes); upstream/ READ-ONLY;
end with a final message stating r6 status + blocker delta + exact FIRE_ORDER path.
