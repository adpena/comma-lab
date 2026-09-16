# ddm_ren1 — record the shipped renderer's training provenance, then price the cheapest refit-in-place on the CURRENT field that keeps the receiver unchanged (charter, MAIN 2026-09-11 host ~23:40Z; operator full-authority GO; Opus)

## Why (the staleness ledger, MEASURED by hpr1's audit `.omx/research/ddm_hpr1_staleness_audit_20260911.md`, rung 3)
Four moves today (45–48) came from counted sections fit to older states of the object (predictor clipped; prior mis-rounded; prior fit to
the move-26 field; the rounding composed on the refit). The audit's rung 3 is the LARGEST counted section: the semantic renderer
(`semantic` member, 29,862 B, sha 786950a5…, byte-identical from the move-39 tree through move 48) whose training provenance is
UNRECORDED (which field version, which losses, which checkpoint/epoch, which quantizer) and which carries a measured prior negative:
ft1 saw +31.23 % d_seg at the first epoch of a fine-tune, rf1/ft1 found the seg↔pose coupling 170–220 (seg-only refit unpayable —
memory `renderer_seg_pose_coupling_170_220_two_arms_20260903`), rw1 found no repair on the INT4 code grid at n600, ntb2 closed 3-bit on
every layer (first PoseNet measurement of renderer depth cells) but explicitly left a QUANTIZATION-AWARE REFIT open. The field moved
eight times since the renderer was fit (31/32/35/38/39/40/42/43). Unlike the prior and the mixer, a renderer refit is NOT
output-lossless: the frames change, so d_seg/d_pose change and the pose re-solve law binds (resolved pose only; frame 0 = repair in
the admission). That is why this charter has two halves and the second is gated on the first.

## Pointer (binding; copy, never edit)
Move 48: S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600]; archive d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c;
promoted tree `/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime` (SEALED — copy). Components d_seg
0.00010345, d_pose 4.59e-6; pose budget ≤ 1.25e-4 (memory m110); admit bar −2e-5 at 179,111 B; exact sqrt per rung (never a linearisation).

## Half 1 — PROVENANCE (read-only, $0, mandatory before any training)
1. From the receiver code, the shipped `semantic` member's parser, and the memos/commits that produced it (grep the store: rw1, rf1, ft1,
   sm3/SM3R, the move-39 packet, mp2 semantic receiver), record: architecture (layers, widths, activations, FiLM), parameter count,
   quantizer (bits per tensor, scale rule), the FIELD VERSION it was trained on (sha of the token field), the losses (seg? pose? both?
   weights), the training data (which pairs; GT lineage PyAV vs DALI), epochs/checkpoint, EMA, and the exact producer commit. Every claim
   with a receipt; unknowns stated as unknown, never inferred.
2. Successor check table: rf1, ft1, rw1, ntb2 (3-bit), sd1 (ancestor seg-only sweep), md1–md4 (persistent partition), ls1/ls2 — what each
   closed, at which scope, on which object; what is NOT closed (QAT refit; per-row mixed depth; MODE_ROW_PRUNE; a refit on the CURRENT field
   with resolved pose).
3. Measure the renderer's CURRENT render floor on the current field at a correct partition (obx2's `decompose_seg_error`; the pre-burn gate
   3.1990e-4) and its d_pose sensitivity to ±0.5 LSB smooth perturbations (obx2's 9.40× noise law: smooth vs noise) — n600, frozen CPU
   scorer, `[macOS-CPU advisory]`. This is the step-0 probe md4 demanded before any burn.

## Half 2 — the cheapest refit-in-place (gated on Half 1; STOP and report if the gate fails)
Gate: Half 1 shows the renderer was fit to an OLDER field AND the measured floor/sensitivity leave room for a refit that keeps the
receiver unchanged (same architecture, same quantizer, same member size or smaller). Then: refit the renderer's weights on the CURRENT
token field with BOTH scorers in the loop (seg + pose through R, resolved pose re-solved after every field change — never seg-only),
warm-started from the shipped weights, quantization-aware in the shipped format, EMA, per-stage checkpoints, eval_roundtrip; price at n600
through the frozen CPU scorer on all 600 pairs (no prefix), exact bytes with the real coder (the member may shrink; report Δmodel and
any Δtail). Fire rule: net ΔS < −2e-5 at exact bytes with d_pose within the budget → cold n600 parse-back (raw will DIFFER — retain it with
sha), manifest outside the tree, smokes, first-measurement inputs (identity class breaks: a new token plane? NO — the field is unchanged;
but the raw changes, so declare the class honestly), STOP for MAIN. Else the closure memo with the numbers and which term ate it.

## Boundaries
Never edit `upstream/`, the PR tree, sealed trees, or other arms' directories (ddm_tmx1 LIVE; ddm_hpr1 stopped but its trees are the pointer's);
$0 until a seal; no Modal; heavy steps through `tools/launch_detached_process.py --output-dir … --nice 0 --done-receipt … -- <cmd>` with resumable
per-stage checkpoints (non-negotiable); bulk under `/Volumes/VertigoDataTier/pact/ddm_ren1/` (Vertigo ≥ 50 GiB; never APDataStore; never the local
disk; never lower a reserve); every payload with sha; MPS allowed as the training-gradient device only, NEVER as a score (frozen CPU scorer for
every number). Memo `.omx/research/ddm_ren1_renderer_provenance_and_refit_in_place_20260911.md` (`# FORMALIZATION_PENDING:<rationale>` or a cite);
checkpoint as `ddm_ren1`; serializer commits, two review passes per .py, no co-author trailer, `[no-triality] [p0-ledger-ok]`.

## OPTIMAL FORM
- Reference form: the renderer's own original training (once Half 1 names it) at full n600 with both scorers; obx2's step-0 probe; hpr1's refit
  discipline (warm start, held-out, real coder). Every delta vs the reference is SCOPE (a refit-in-place, not a new architecture); a seg-only or
  prefix-scored run is a TOY and cannot produce a verdict.
- Provenance pins: hpr1's audit (sha eaac1307…); the memos Half 1 finds; move 48 tree/archive; obx2's closure memo (98c97390…).

## Prior negatives accounted (operator 2026-08-15)
- rf1/ft1 coupling 170–220: seg-only refit unpayable — both scorers in the loop, resolved pose, or nothing.
- rw1: INT4 code-grid repair closed at n600 — you refit weights within the format, you do not search the grid.
- ntb2: 3-bit closed on every layer; QAT refit explicitly open — this is that door.
- obx2: render floor at a correct partition must be measured BEFORE burning (pre-registered gate 3.1990e-4; the shipped renderer's own floor
  is the pointer's d_seg 1.03e-4 at exact tokens — state whether a refit can lower it or only hold it while shrinking bytes).
- md4: the free step-0 probe predicts unreachability at 9 in 10 — run it first; a burn that skips it is the forbidden shape.
