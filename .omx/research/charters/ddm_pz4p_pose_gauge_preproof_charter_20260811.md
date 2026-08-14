# ddm_pz4p — pose-gauge QAT PRE-PROOF: convert pz4's fire gate into a measured receipt

## Mission (operator 2026-08-11 "Research and believe" × the pose axis = 91.3% of the own-vehicle gap)

fd135 (commit 60ec8c21b0) queued `learned_pose_gauge_qat` with owner pz4 and fire trigger:
"proposal pre-proves at least 2,000 B savings and MSE below 2.5e-6." Nobody has attempted the
pre-proof. Your job: DESIGN the learned pose-gauge quantization (per the operator quantization
toolbox law: ADAPTIVE per-cell · AWARE in-loop · sub-int16 depth) and MEASURE the pre-proof
SCORER-FREE — reconstruction MSE against BANKED exact PoseNet reference outputs + real-coder
byte count vs the shipped pose carrier. Deliver the pre-proof receipt (fires pz4) or an honest
refusal with the measured (MSE, bytes) frontier.

RECALL FIRST: `tools/corpus_query.py` over {pose gauge, pose carrier, CPR1, pk2, dxi, low-rank
pose codec, semantic pose} + read: pk2 receipts (PR130 pose-carrier representation attack —
23,384 B / 12.24% / 0.0155704 S-contribution) · ps135 gen-1 memo (carrier dims, SD1M rungs,
compensation owed) · #140 low-rank rank-2 SVD pose codec (2.7× cut at MSE ≤ d_pose) · the
quantization toolbox memory · fd135's pose sections. The banked PoseNet outputs live in the
DT1/ps135/pk2 retained stores — verify custody (sha) before consuming; if absent, DERIVE the
reference from retained frames is FORBIDDEN scorer work — instead name the missing custody and
measure what IS banked.

## Ordered work

1. **CUSTODY:** locate + sha-verify the banked exact pose reference outputs and the shipped
   pose-carrier bytes on the live lineage (lc2/cp135). State exactly which reference you
   measure MSE against — the pre-proof gate is defined on that surrogate, say so.
2. **RESEARCH LEG:** learned-gauge/QAT literature for tiny structured payloads (per-cell scale
   adaptation, quantization-aware fitting, low-rank × quantized composition with #140's rank-2
   result) + PR133/PR135's own quantize-then-COMPENSATE mechanism (m05: CBQ alone 29× worse —
   compensation is mandatory; your gauge must include the compensation solve, not raw CBQ).
3. **BUILD + MEASURE the pre-proof:** the gauge candidate(s) fitted to the real carrier, MSE vs
   banked reference, bytes via the real shipped coder path. Sweep the (depth × per-cell × rank)
   grid to the measured (MSE, bytes) frontier — not one point.
4. **DELIVERABLE:** pre-proof receipt {candidate, bytes_saved vs shipped, MSE, custody shas}
   OR honest refusal w/ the frontier curve; durable memo
   `.omx/research/ddm_pz4p_pose_gauge_preproof_20260811.md` w/ NEXT_IF_RESUMED; if the gate
   passes, emit the pz4 fire-ready row (owner MAIN routes the full pz4 arm).

## Boundaries

Scorer-FREE hard boundary: NO PoseNet/SegNet forward passes (ps135b owns the scorer lane);
MSE only against BANKED reference outputs with verified custody. No Modal. Serializer commits
post-edit sha, [no-triality] [p0-ledger-ok], --no-co-author. Payload law: every candidate
gauge payload retained w/ sha256+bytes (both SSD tiers granted). First-attempt-too-slow ≠
verdict: if a fit is slow, attribute then optimize before any close.

## OPTIMAL FORM

Reference form pinned: shipped pose carrier on lc2 sha
f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45 (187,226 B archive) + pk2's
measured 23,384 B pose section · PR135 mechanism receipts (fd135 commit 60ec8c21b0; ps135 gen-1
memo commits d6ea363904/452ce890b5/f5615682e8). SCOPE = the full real carrier (all 600 pairs'
pose stream), never a pair subset for any cited number. PRIOR-LAW PREDICTION (derived fresh):
#140 measured 2.7× at MSE ≤ d_pose via rank-2 SVD alone; composing rank reduction WITH
per-cell adaptive sub-int16 gauge + PR133-style compensation predicts ≥2,000 B (≥8.6% of the
23,384 B section) at MSE ≤ 2.5e-6 — the gate PASSES; FALSIFIER: the measured (MSE, bytes)
frontier never touches the (2.5e-6, −2,000 B) box → pz4 closes REFUSED-BY-PREPROOF with the
curve as the deliverable, and the pose axis rests entirely on ps135b/js1.
