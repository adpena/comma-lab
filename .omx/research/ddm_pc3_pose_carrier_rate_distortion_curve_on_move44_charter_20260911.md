# ddm_pc3 — the pose-carrier rate/distortion curve on move 44's field: price d_pose against carrier bytes with the real re-solve and the real coder; seal the S-optimal point (charter, MAIN 2026-09-11; operator full-authority GO)

## Why this is the highest-marginal untried lever
Pointer move 44: S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600]; d_seg 0.00010345 (term 0.010345), d_pose 4.59e-06 (term
sqrt(10·d_pose) = 0.006775), rate 0.120125. The pose term's MARGINAL value is 5/sqrt(10·d_pose) = **738 S per unit d_pose**, 7.4× seg's 100.
Halving d_pose (4.59e-6 → 2.3e-6) is −1.98e-3 S = 99 bars, worth up to +2,970 B of carrier at the exchange 6.6586e-7 S/B. Every pass so
far RE-SOLVED the carrier at its existing lattice (sj1 pass 6: carrier +8 B, d_pose 4.649e-6 → 4.587e-6; `.omx/research/ddm_sj1_t4_token_predistortion_pass6_20260910_pointer_move_43_20260910.md` sha d57ab4d55b3d7b0d…), never
priced the d_pose-vs-carrier-bytes CURVE. pc1/pc2 (memory) measured that carrier bytes buy a POSITIONED SUBSPACE / a lattice point, rank cut
free in span but 4–2,733× costly on the lattice; mc1 closed the motion-compensated plane. Nobody has swept the carrier's rank/lattice/precision
with the real re-solve and priced each point by real encode + frozen-scorer d_pose at n600. Do that, on the SHIPPED move-44 field.

## Deliverable
1. Read the incumbent's carrier: `cpr1/carrier_codec.py`, `runtime/carrier_repack.py`, the pose re-solve used by sj1 pass 6
   (`experiments/ddm_sj1_*` carrier chain; `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/carrier_chain/` receipts) and the move-44 tree
   (`/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime`, read-only; copy). State the carrier's exact parametrization
   (rank, lattice step, precision, per-pair vs shared) and its counted bytes today.
2. **The curve.** Sweep the carrier's capacity axes (rank ×2 and ×4; lattice step ÷2; precision +1/+2 bits; per-pair terms for the 60 worst
   pose pairs only) with the REAL re-solve on the SHIPPED field (token planes unchanged), n600, frozen CPU scorer for d_pose [macOS-CPU advisory]
   (pose base measured on the pointer's own configuration on YOUR instrument first — the pose-base law), each point priced by the REAL encode
   of the whole archive (twin encodes). Report per point: carrier bytes, archive bytes, d_pose, d_seg (must be unchanged: token planes fixed
   — verify), projected S, net vs the −2e-5 bar. Keep every payload.
3. **Seal the S-optimal point** if it beats the bar: full cold n600 public parse-back, raw identity of the token planes vs move 44's retained
   raw (only the carrier-driven frames may differ; state exactly which bytes differ), manifest regenerated outside the tree, literal census,
   smokes, decode wall-clock inherited (receiver unchanged → t4_direct inheritance from move 44's leg) → `tools/make_candidate_seal.py`
   (candidate_seal.v2 path: the receiver is UNCHANGED, so this is a normal seal, not a first-measurement intent). MAIN fires.
4. Memo `.omx/research/ddm_pc3_pose_carrier_rate_distortion_curve_on_move44_20260911.md` with the curve table, the chosen point, the seal
   path, falsifiers pre-registered (pose-base ratio, d_seg identity, twin identity, raw identity outside carrier frames). Serializer commits
   (two review passes per .py; `[no-triality] [p0-ledger-ok]`; no co-author trailer). Checkpoint as `ddm_pc3`; mark COMPLETE at the end.

## Boundaries
No Modal (MAIN fires), no receiver code change (a carrier FORMAT change = receiver change → first-measurement contract; STOP and say so if the
best point needs one), never edit `upstream/`, the PR tree, sealed trees; n600 only; every payload on the SSD tier with sha; heavy steps
through `tools/launch_detached_process.py --done-receipt …`; do not touch obx2/nt1/mx1 directories.

## OPTIMAL FORM
- Reference form: sj1 pass 6's carrier re-solve chain (the landed instrument) and its pose-base measurement; the real coder for every price;
  a reduced sweep grid is SCOPE (declared), a proxy scorer or proxy coder is MECHANISM (toy-bracketed).
- Provenance pins (sha256 prefixes): move-44 packet memo f7638e1e171e0d19…; move-43 packet memo d57ab4d55b3d7b0d…; pc1/pc2 memories (record their shas); pointer
  commit 99625f32f / archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e; the shipped field subset6.u8 (a92e7d90…, from
  the encode receipt).

## Prior negatives accounted (operator 2026-08-15)
- pose-base law (m: POSE BASE = pointer's config on the arm's instrument): measure the base first; borrowed bases flip the sign.
- pc2: rank cut is free in span but costly on the lattice — sweep BOTH axes; do not assume a byte buys the same d_pose everywhere.
- mc1: the motion-compensated previous plane is closed; do not re-open it as "temporal carrier".
- POST-HOC CHANGE OF AN IN-LOOP CHOICE = PAID AT SCORER: the re-solve must be the in-loop solver, not a post-hoc fit.
- container-break lottery (sd 34.8 B): price by real twin encodes, ship the smaller twin, never search the container.

Final message: the curve (one line per point), the chosen point, seal path + sha, falsifiers, and the frontier line
`composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)` unchanged (MAIN fires).

<!-- # FORMALIZATION_PENDING: producer charter; the curve becomes an equations-leg law with the exact row -->
