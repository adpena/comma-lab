# CHARTER — ddm_pz4r_full_n600_eval (2026-08-13)

Run the ONE retained full-n600 public-runtime evaluation that decides PZ4R — the strongest
unconsumed pose lead per the hv1 fresh-eyes review (memo sha 1e071f66…, its rank-2 queued
fire-order). PZ4R is a receiver-closed 183,137 B recode of the cp135 base (MEASURED −4,089 B
rate saving, scorer-free build) whose DISTORTION has never been measured. One evaluation
resolves it: if d_seg and d_pose hold at base values, PZ4R is a −4,089 B ≈ −0.00272 S rate
move ≈ 22.8% of the remaining −0.011955 gap; if distortion regresses past break-even, the
object is closed at instance scope.

## OPTIMAL FORM
Measurement arm, not a build (OPTIMAL_FORM_NA: single evaluation of an existing receiver-closed
artifact through the established local exact protocol — no mechanism reduction possible; the
protocol IS the family reference form). Full n600, never a subset. Provenance pins REQUIRED:
archive path + sha256 + bytes recorded before decode; every claim MEASURED-labeled with axis.

## FIRE-TRIGGER CHECKLIST (hv1's conditions — satisfy ALL before running)
1. Retained storage plan: preflight free bytes on /Volumes/VertigoDataTier (raw decode ≈ 3.7 GB
   + fields ≈ 120 MB + reserve); output under
   /Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval/.
   ALWAYS KEEP THE PAYLOAD (P0 DEF CON 1000): retain decoded raw, argmax field, pose vectors —
   sha256 + bytes in the result JSON. Certify-or-block; never delete.
2. Free scorer lane: verify no live heavy scorer job (governed admission; check running trainers).
3. Explicit claim: tools/claim_lane_dispatch.py claim --lane-id ddm_pz4r_full_n600_eval
   --platform local (terminal row on completion).
4. Physical lock: exclusive run-lock file in the output dir (the sa1 concurrent-resume lesson).

## THE MEASUREMENT
Evaluate the PZ4R candidate archive at
/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/ through the
SAME entry point a remote would run (its inflate/receiver front door — the po1 Round-2 lesson:
local parse-back that bypasses the runtime's own verification is NOT custody). Then frozen
CPU-torch SegNet argmax field (n600, full population) + PoseNet first-6 vectors vs the retained
GT references (the js1b retained fields at /ddm_js1b_retained/… have T4 provenance — if reusing
local CPU references instead, LABEL the axis honestly: [macOS-CPU advisory] vs [contest-CUDA]
comparisons are DIFFERENT instruments; never mix without saying so). Emit: candidate flips vs
base 34,970 · d_pose vs base 6.885642960696714e-6 · exact archive bytes vs 186,252 · joint ΔS
arithmetic with ALL THREE terms + relative share of the −0.011955 gap.

## OUTPUT
.omx/research/ddm_pz4r_full_n600_eval_20260813.md — the measured row, honesty labels, joint-ΔS
verdict with verdict_scope declaration on any negative, follow-ons FIRED/FOLDED/QUEUED same-turn,
NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS. Commit via the serializer (post-edit shas). NO
Modal dispatch (a T4 confirmation row, if earned, is MAIN's fire). Resumable: per-stage
checkpoints; a crash must resume from disk.
