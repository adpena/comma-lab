# ddm_pz4a — $0 pre-proof: sensitivity-allocated variable-precision pose-coefficient recode

Tags: [no-triality] [p0-ledger-ok]  ·  Axis: [scorer-free coder measurement + derivation]
Score claim: false. PLANNING-BAND labeling throughout (pass-3-parent objects: stale for terminal
BINDING per m37, admissible for planning arithmetic when labeled).

## MISSION
js1 Amendment 10 names a T7 rate-race candidate: variable-precision recode of the (600,12) int16
pose coefficient stream, depth allocated per-dim/per-pair from the solve's sensitivity map, with
compensation. It inherits the pz4 2,000 B PRE-PROOF GATE: prove a projected ≥2,000 B net saving
BEFORE any build fires. Produce that pre-proof honestly — or refute it. The likely refutation
mechanism (state it, test it): the shipped entropy coder ALREADY exploits low-magnitude/low-
variance dims, so raw-domain arithmetic wildly overstates the win. The comparison baseline is the
SHIPPED CODED size of the coefficient section, never raw 14,528 B.

## PRIOR-LAW PREDICTION (m38, falsifiable)
The raw-domain saving projects large (≥8 KB) but the CODED-baseline saving lands under 3 KB —
near the pz4 gate, sign uncertain. If the coded saving clears 2,000 B cleanly, that is the
finding; if it refutes, the candidate is dropped at T7 with this receipt and no build is owed.

## INPUTS (verified present; read-only except your own outputs)
- Coefficients: rehearsal-store `(600,12)` int16 NPY, 14,528 B, SHA `2daec0ae99e8...` at
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/t0_rehearsal_pass03/` (t0r1 receipts
  name exact paths). PLANNING parent = pass-03 selected archive `93f8d7b4b...`.
- Sensitivity map: pass-level `(600,6,12)` Jacobian + `(600,3)` active-dims chunks (pass-02
  PARENT — planning only, say so in every table).
- Shipped coded baseline: extract the coefficient section's ACTUAL coded size from the selected
  archive via the lossless CX2/TM1 receiver parse (t0r1's adapter chain) and/or fd135's
  decomposition receipts. If the section is coded jointly with others, report the joint cell and
  derive an attributable bound honestly — never invent a per-section number the container does
  not define.

## METHOD (all local, no scorer, no GPU)
1. Reproduce the baseline: code the raw stream with the SAME coder family the container uses
   (F26/HPAC path if reachable; else brotli-Q11 + LZMA1 as declared proxies, labeled) → baseline
   coded size B0. Cross-check against the shipped section size.
2. Derive per-dim depth from the Jacobian: per-dim sensitivity summary (|J| distribution across
   600 pairs) → reverse-waterfill depth allocation at a pose-contribution tolerance ladder
   (use the measured pose contribution ≈0.0083-0.0098 band as the operating region; NO invented
   tolerance — sweep a small ladder and report the curve).
3. Recode at each ladder rung (quantize per allocated depth; store the depth map's own bytes —
   the allocation metadata is COUNTED) → coded size B(rung). PERSIST every recoded stream +
   sha256 (ALWAYS KEEP THE PAYLOAD; retain under
   /Volumes/APDataStore/pact/ddm_pz4a/retained/).
4. The pre-proof number: max over rungs of [B0 − B(rung) − metadata] with the induced coefficient
   ERROR reported per rung (max |Δcoeff| per dim, in quanta) — the distortion side is NOT scored
   here (scorer-free); report the error so T7's compensation stage knows what it must recover.
5. Verdict row: {CLEARS_2000B_GATE / NEAR_GATE_SIGN_UNCERTAIN / REFUTED} + the honest mechanism.

## OPTIMAL FORM
Family reference: $0 coder-race/probe arm (lp135 4811eb3a is the coder-measurement reference).
Scope-reductions (legal): planning-parent objects; declared coder proxies if F26 path unreachable
(label which). Mechanism-reductions: NONE — real coders on real bytes, no synthetic streams.

## BOUNDARIES
- SCORER-FREE (sole lane = ps135b). No SegNet/PoseNet/evaluator/MPS/Modal/GPU.
- Read-only on the live solve tree + rehearsal store; outputs → APDataStore ddm_pz4a/.
- On BLOCKED-GIT: untracked artifacts + NEXT_IF_RESUMED fire-order w/ exact serializer command;
  MAIN handoff-commits. Serializer w/ --no-co-author + post-edit shas; REVIEW_GATE_OVERRIDE .md only.
- RECALL EVIDENCE mandatory (incl. lp135's coder-closure receipts — do NOT re-litigate the
  lossless same-state coder race; this arm changes the REPRESENTATION, which is the one move
  lp135 left open). End with NEXT_IF_RESUMED + DEAD-ENDS + the standing frontier line:
  own-vehicle lc2 S 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]; effective floor
  cp135 composed 0.16195513827824176 @ 186,252 B (ours).
