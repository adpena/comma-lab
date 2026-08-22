# CHARTER ddm_jo2_solve_reseal (successor of ddm_jo1_joint_objective_design; gs3 rank-2, the seg+pose joint door for sub-0.12 #1182 (memo .omx/research/ddm_r012_rate_representation_20260821.md))

GOAL: drive the jo1 SOLVE seal (/Volumes/APDataStore/pact/ddm_jo1_joint_objective_design/seal_r4/, READINESS status=BLOCKED) to READY_TO_FIRE_UNDER_STANDING_GO. NO heavy launch from this arm — MAIN fires the governed slot.

## RECALL FIRST (stores, not working memory)
- seal_r4/READINESS.json — 5 blockers, workload sha ef3134ce1a60188912463dc213f5d79c5d3db1d6ba9bdde5f2f551a9c88a72ab.
- /Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/harvest_r8/MATERIALIZER_RESULT.json — r8 Modal materializer COMPLETE/MATERIALIZED; deterministic repeat byte-identical (0.raw sha 6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883).
- Local payload custody payloads_r8/: fx5_e1_argmax_n600.npy sha256 e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34 (117,964,928 B) + fx5_e1_first6_n600.npy sha256 71f7d2639eb624f4d0eb89e40ac5956a74b1f72951dc7f07424468769af8350f (14,528 B). Bulk seg logits/inputs stay on Modal volume comma-auth-eval-cache-artifacts under ddm_jo1u_fx5_e1_n600_r4 (shas in the receipt).
- qs5 in-compile Schur compensation PROVEN (#1042, memo .omx/research/ddm_qs5_verdict_and_no_toy_enforcement_20260813.md: d_pose BELOW base, repeat identical) — the mechanism to port.
- fx5_e1 decode is BYTE-IDENTICAL to rc2 (fx5 = pure rate −70 B; d_seg/d_pose unchanged), so the materialized argmax/pose6 fields ARE the rc2 fields.
- .omx/research/ddm_wd4_warm_lineage_width_20260821.md advisory chain — the pin-bound-runtime lesson (#1123 genus, memo .omx/research/ddm_wd4_warm_lineage_width_20260821.md).

## PRIOR-LAW PREDICTION
Per #1042 the in-compile compensation ports cleanly (exact solve, not fitted). Per the r4 reseal lesson the ArtifactRef pin is a TRIPLE (sha256 + source_object_sha256 + bytes) — update ALL THREE from the working tree, never partially (partial updates cost 3 PIN_DRIFT rounds on r8).

## WORK, in order
1. Re-point seal inputs at the materialized payloads (blockers RC2_BASE_ARGMAX_FIELD_MISSING + SOURCE_POSE6_TARGETS_MISSING → resolve against payloads_r8 + the existing DALI GT Pose6 table from the qs1/#1142 wiring (memo .omx/research/ddm_na10_* DALI-GT reopened queue); verify shas at consumption).
2. IMPLEMENT RC2_FRESH_SCHUR_RECEIVER_CLOSE: port the qs5 proven in-compile compensation onto the fx5_e1/rc2 body receiver-close path, asserted IN CODE (compensation computed fresh per candidate — the qs4 stale-compensation disaster is the named anti-pattern).
3. AP_STORAGE_PREFLIGHT (free=33.4 GB vs required=47.2 GB): EITHER re-root solve scratch to local APFS (scratch legal locally; the ExFAT AppleDouble hazard #1122 (memo .omx/research/ddm_wd4_warm_lineage_width_20260821.md advisory chain) also argues for APFS) OR certify-and-move reclaim on AP (never delete).
4. Run the memory preflight at the REAL config → receipt.
5. Reseal → READINESS READY_TO_FIRE_UNDER_STANDING_GO with derived wall-clock + blockers [].

## OPTIMAL FORM
Provenance pin: experiments/ddm_jo1_joint_objective_design.py=f787f53a39239bbeb4f27518873fca2100f8fca3c70e5219c8df28e1b351deda (the reference StageConfig contract lives in this file at this sha).
Reference form = jo1's own StageConfig contract (REQUIRED_STAGE_IDS target_birth/joint_balance/collateral_finish in experiments/ddm_jo1_joint_objective_design.py). SCOPE reductions (bounded smokes) legal; MECHANISM reductions forbidden (no proxy scorers; no subset verdict cited as a finding — n600 or labeled non-evidence). Payloads persisted per P0 ALWAYS-KEEP-THE-PAYLOAD (sha256 + bytes in every receipt).

## PROVENANCE PINS
- experiments/ddm_jo1_joint_objective_design.py=f787f53a39239bbeb4f27518873fca2100f8fca3c70e5219c8df28e1b351deda
- experiments/ddm_jo1_modal_joint_objective.py=84290ab0fbcd735cdac7c7b49a6ee4dfd3cd50ffc580e2e89410e7503103e7cc
- seal_r4 compiled workload sha ef3134ce1a60188912463dc213f5d79c5d3db1d6ba9bdde5f2f551a9c88a72ab

## CONTRACT
Commits via tools/subagent_commit_serializer.py with post-edit --expected-content-sha256; .py = 2 genuine review passes; durable memo .omx/research/ddm_jo2_solve_reseal_20260821.md; final message states READY-or-blocked with named blockers. Read CLAUDE.md + AGENTS.md first.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends, accounted)
- qs4 REFUSED +2.437870e-4 (stale Schur compensation carried cross-object — memo
  .omx/research/ddm_qs5_verdict_and_no_toy_enforcement_20260813.md §history): THE anti-pattern
  this arm's step 2 exists to make unrepresentable (compensation asserted fresh IN CODE).
- qs5 R2 REFUSED +2.519822e-6 near-miss (same memo): compensation itself PROVEN (pose below
  base), the refusal was seg/rate economics — the mechanism ports, the economics must be re-won
  on the jo1 joint objective.
- pk4 FORMULATION ceiling (memo .omx/research/ddm_pk4_optimal_form_frame0_pose_verdict_20260813.md):
  linear per-pair frame-0 overlays dead at 43–997 B — the jo1 joint solve is the NONLINEAR/joint
  route those verdicts route pose→0 through; do not re-open linear overlays.
- wd4 GATE-FAIL 1,792× over bar (memo .omx/research/ddm_wd4_warm_lineage_width_20260821.md):
  the slice-warm-start rate door is closed at instance scope — jo1 is now the primary
  rate-representation door; its failure modes (pin drift, ExFAT sidecars) are documented there.
