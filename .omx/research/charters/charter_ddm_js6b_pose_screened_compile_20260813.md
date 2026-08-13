# CHARTER — ddm_js6b_pose_screened_compile (2026-08-13)

CONSUME THE FAMILY LAW (`.omx/research/ddm_re1_round1_full_auth_row_20260813.md`): on the
cp135 base, unprojected HP3 semantic-cell edits are POSE-DOMINATED — measured per-cell pose
damage 5.7e-6…3.4e-5 S (2 independent instances: re1 Round-1, JO1) vs per-flip seg value
8.477e-7 S, through the SHARED resize D (ddm_pz1: PoseNet + SegNet consume the identical
`interpolate` output). The seg EXISTENCE proof stands (−2 flips realized at the exact T4
instrument, receiver-null refuted). The missing piece is SELECTION: pose-screening before
compile. Scorer-free build; NO Modal fire (MAIN fires); local only.

**OPERATOR DOCTRINES BINDING:** "no naive or toy or generic basis ever" (screens DERIVED
from measured leakage + Q3 machinery, never a guessed threshold) · byte-closed-row cadence
(output = ONE sealed batched fire-order producing a complete row) · "as much as possible
locally" (only the CUDA-locked decode + scorer forwards leave the Mac).

## OPTIMAL FORM
Build arm (OPTIMAL_FORM_NA: no mechanism raced; reference forms = the PROVEN re1t/sa1 T4
worker `experiments/ddm_re1t_modal_t4_sign_gate.py` lineage + the #837 Q3 pose-null
machinery + js6's proposal bank). ADAPT, never rebuild. Provenance pins on every consumed
component.

## THE BUILD (3 legs)
1. **Worker extension — pose vectors in the SAME dispatch.** Extend the re1t T4 worker to
   ALSO retain PoseNet official 6-vectors per pair (all 600) alongside the SegNet argmax
   field — PoseNet forward on the already-decoded frames is ~free on T4. One dispatch =
   BOTH axes; no more seg-only provisional gates. Retain per the payload law; byte-identical
   worker behavior when the flag is off.
2. **Pose-screen js6's 200 proposals** (`/Volumes/VertigoDataTier/pact/
   ddm_js6_seg_representation_join_20260813/proposal_bank`): per proposal, estimate the
   pose interaction using (a) Q3 pose-null projection of the cell edit (#837 machinery —
   the exactly-pose-null frame_1 subspace, measured seg-reachable) and/or (b) the measured
   leakage priors (603 S/unit d_pose marginal; the 2 measured instances as calibration).
   Rank by screened net ΔS = seg value − pose-risk bound. Proposals whose SCREENED bound
   cannot beat 0 are held, not compiled. State the screen's honesty limits explicitly —
   it is a local prior, not the exact instrument; the T4 dispatch is the verdict.
3. **Batched compile + sealed fire-order.** Compile the top pose-screened proposals
   (batch the survivors into ONE candidate archive where composable, else the single best)
   through the HP3/RC64 closure → byte-closed candidate + adapted runtime (po1 pin lesson:
   inflate pins the CANDIDATE archive sha) → sealed fire-order JSON (fresh run-id, exact
   command, ~$0.16, #381 ~$2.7 spent) for the EXTENDED worker (seg field + pose vectors in
   one dispatch). MAIN fires. Admission rule pre-encoded: net realized ΔS < 0 on BOTH axes
   measured, per hv1 (ONE whole candidate, complete recomputed S — no projection claims).

## OUTPUT
`.omx/research/ddm_js6b_pose_screened_compile_20260813.md` + code/tests + the sealed
fire-order at a fresh consumer store. Commit via `tools/subagent_commit_serializer.py`
(post-edit shas, `[no-triality] [p0-ledger-ok]`). End with NEXT_IF_RESUMED +
LIVE-HYPOTHESES + DEAD-ENDS.
