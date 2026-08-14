# ddm_pz4r — the PGQ1 receiver: realize the passed pre-proof (pz4p's FIRED routing row)

## Mission (pz4p FIRED-TO-MAIN-ROUTING, trigger SATISFIED: 19,221 B + MSE 1.0986e-6)

pz4p (commit 749f4677f8, memo `.omx/research/ddm_pz4p_pose_gauge_preproof_20260811.md`) PASSED
the pz4 fire gate 9.6× over: winner `r6_b12_global`, PGQ1 gauge 3,837 B replacing the ~23.4 KB
CPR1 pose carrier, exact rate envelope 168,005 B (−19,221 B vs LC2), banked-output MSE
1.0985637375134246e-6 < 2.5e-6. NOTHING IS REALIZED YET: no receiver parses PGQ1, no frame was
rendered, no scorer ran. Your job is the arm pz4p named: **build the resumable receiver that
consumes PGQ1 and removes CPR1**, byte-close the new archive, and measure the REALIZED
(d_seg, d_pose) through the real decode. Consumer store:
`/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/`.

RECALL FIRST: pz4p memo + its retained `preproof_v3` store (330 receipts, payloads
sha-verified) · pk2 receipts (the CPR1 mechanics being replaced) · ps135 gen-1 memo (carrier
dims, receiver seams) · lc2 receiver/inflate sources (submission dir) · the realization-gap law
([[realization_gap_is_fixable_through_actual_S_R_GT_20260806]]): a surrogate MSE is NOT a
realized d_pose until it survives the actual decode.

## Ordered work (staged; scorer LAST and lane-gated)

1. **RECEIVER BUILD (scorer-free):** extend the lc2 receiver to parse PGQ1 (r6_b12_global) and
   reconstruct the 600×6 pose object; remove CPR1; deterministic, resumable, repeat-identical
   builds. First rung = DIRECT consumption (no joint training) — measure what the passed gauge
   realizes as-is before any training rung.
2. **BYTE-CLOSE (scorer-free):** new archive w/ exact parse-back + repeat-identity; expected
   ≈168,005 B (reconcile any delta vs pz4p's envelope explicitly). Retain EVERYTHING
   (payload law; both SSD tiers granted).
3. **REALIZED MEASUREMENT (scorer-gated):** full n600 d_seg + d_pose through the REAL decode
   [macOS advisory axis]. The scorer lane is owned by ps135b — claim it ONLY when free
   (poll `codex_arm_queue.py status`); if still held at your build completion, land the
   built+byte-closed state with a QUEUED-WITH-FIRE-ORDER row for the scorer pass and exit
   clean. NEVER run a scorer beside ps135b.
4. **BOTH AXES, ALWAYS:** report seg collateral (changed frames touch seg) beside pose — a
   pose-only report is the named half-measurement anti-pattern. If direct consumption degrades
   the surrogate, the JOINT-TRAINED conditioned receiver (pz4p live-hypothesis 1) is the next
   rung — design it, do not fire training in this arm.
5. **STAGE 2 (design-only):** cross-lineage refit sketch for the cp135/PR135 base (pz4p
   live-hypothesis 2, transfer unmeasured) — a fire-order row, not a build.

## Boundaries

No Modal. Scorer only via a free lane claim (rule above). Serializer commits post-edit
--expected-content-sha256, [no-triality] [p0-ledger-ok], --no-co-author. Durable memo
`.omx/research/ddm_pz4r_pgq1_receiver_20260811.md` w/ NEXT_IF_RESUMED. Honest fail-closed
close if the receiver cannot reach parse-back parity — name the seam. First-attempt-too-slow ≠
verdict (attribute, then optimize, before any timing-based close).

## OPTIMAL FORM

Pins: pz4p commit 749f4677f8 · winner r6_b12_global · PGQ1 3,837 B · envelope 168,005 B ·
MSE 1.0985637375134246e-6 · base lc2 archive sha
f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45 (187,226 B, contest-CUDA
0.16959899569230852) · preproof_v3 store sha-verified receipts. SCOPE = full n600 real decode;
no pair subsets for cited numbers. PRIOR-LAW PREDICTION (derived fresh from the surrogate
arithmetic): direct PGQ1 consumption realizes joint ΔS on lc2 in **[−0.0095, −0.0124]**
(rate −0.012798 from −19,221 B; pose penalty between the best case Δd_pose≈MSE→+0.0004 and the
triangle-inequality worst case √-additive→+0.0033; seg collateral assumed ≈0 because the frames'
pose steering changes are sub-quantum — MEASURE it). FALSIFIER: realized d_pose > 4e-5 through
the real decode OR seg collateral > +0.002 S → the surrogate-realization gap is the finding;
route to the joint-trained conditioned receiver rung, do NOT close the family.
