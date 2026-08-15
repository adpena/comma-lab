# ddm_av2_fresh_eyes_and_distillation_reopen — adversarial pass on the refusal chain + the distillation reactivation design

## MANDATE (operator 2026-08-15, verbatim steers: "Perhaps another fresh eyes pass is called for" + "Also perhaps distillation is not dead")
Two coupled jobs over the post-hv1-pointer-move / post-wd2-refusal state. $0, no launches, no
Modal. You are FRESH EYES: re-derive from receipts, never trust memo prose.

### MANDATE A — adversarial review of the refusal + pointer-move chain
Object under review: .omx/research/ddm_wd2_ep60_advisory_refusal_verdict_20260815.md (+ its
receipts under /Volumes/APDataStore/pact/ddm_wd2_width_distillation/) and
ddm_hv1_pointer_move_and_wd2_advisory_chain_20260815.md. Seeded hypotheses (audit each,
then sweep BEYOND the seeds):
1. CROSS-INSTRUMENT DELTA (MAIN self-caught): the refusal compared student-on-mirror-CPU vs
   base-on-T4. The same-instrument base leg is NOW IN FLIGHT — receipt lands at
   /Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/contest_auth_eval.json
   (~35 min from 2026-08-15T20:2xZ). CONSUME IT: recompute Δd_seg/Δd_pose same-axis; does
   the 8.2×-over-bar verdict survive? (Magnitudes suggest yes; verify, don't assume.)
   Also relay the base row to ddm_mp2 (its admission rule needs it) via your memo.
2. CROSS-QUANTITY SLOPE TRANSFER: the "160+ epochs" continuation estimate transferred a
   log-slope from train_decode_mse_normalized onto decode_mse_uint8 (only ep1+ep60 endpoints
   exist for the latter). Grade the estimate {SOUND/UNSOUND}; if unsound, what IS derivable
   from the retained data?
3. POSE 0.092 EVIDENCE GRADE: measured under harness stamp "auth-eval env mismatch advisory".
   Is the 634× pose read robust to that grade, or does any known env-mismatch mechanism
   (torch version, thread count, batch shape — the et4 law) plausibly inflate it?
4. MIRROR INTEGRITY CLASSES: two contamination classes now measured (ExFAT AppleDouble `._*`;
   NEW: runs write __pycache__ INTO the mirror — cured 2026-08-15 by sweep +
   PYTHONDONTWRITEBYTECODE=1 in the launch env). Recompute the mirror snapshot sha post-sweep
   (tac.contest_compliance path) and record it vs the d5bb36a2… birth sha; enumerate any
   OTHER mutation vector a run has into the mirror (fail-closed list).
5. mp2 ROUTING OPTIMALITY: given the wd2 refusal mechanism (decode-MSE loss, not capacity per
   se), is rfo2's route order still optimal? Any cheaper rung skipped?

### MANDATE B — "distillation is not dead": the reactivation design (operator steer)
The refusal was INSTANCE-scoped by design; the operator elevates the family from parked to
actively reopened. Deliver a SEALED successor design (wd3) at CHARTER-TIME OPTIMAL FORM:
- MECHANISM (the fix the data demands): SCORER-AWARE distillation loss — student matches the
  teacher's realized SegNet margins/argmax + PoseNet 6-dim outputs THROUGH R (resize→uint8→
  scorer preprocess), not raw decode MSE. This is the campaign's task-space doctrine applied
  to the student; PR-winner precedent: Quantizr KD (kl_on_logits T=2.0). Derive the loss
  composition (decode-MSE anchor term + seg-margin term + pose term; weights DERIVED from the
  S-arithmetic exchange rates, never guessed).
- ASSETS (all retained, payload law): teacher cache 1.83 GB sha 695023d4… (camera renders —
  extend with cached teacher scorer outputs, computed ONCE) · ep60 checkpoints (warm-start
  candidate — adjudicate warm vs fresh under the changed loss; cite #816 fresh-vs-warm) ·
  both advisory work dirs.
- ARCHITECTURE ARMS (pre-registered in the wd2 build arm's LIVE-HYPOTHESES — re-read
  .omx/tmp/codex_runs/ddm_wd2_width_distillation_build.last.txt): dense d4/w56 ("changes the
  inherited computation least") vs factorized d4/w64/r19 vs wider-w96 — with the RATE-PRIZE
  EROSION TABLE: student bytes vs the Δd_seg ≤1.07e-4 + pose-held admission bar; a student
  only pays if (rate prize) + (distortion cost) < −3.5e-6 net on the hv1 base.
- VALIDATION IN-LOOP (fd2 lesson): periodic realized verdicts through the REAL chain (R +
  uint8 + frozen scorers) on a fixed pair subset — never train blind for 60 ep again; and a
  STRIDED (not prefix) subset per m88/m96.
- Deliverable: wd3 charter + config, gated on Mandate A item 1's same-instrument base row;
  MAIN fires the Metal slot when r5 (pid 63183) exits. NO launch from this arm.

## OPTIMAL FORM
Reference: the wd2 build arm's landed trainer/exporter harness (re-derive paths from its
.last.txt + TRAIN_RESULT.json custody) + the proven advisory chain (mirror + shim + launcher).
Provenance pins: hv1 frontier archive sha
80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e @182,759 B · mirror birth
snapshot sha d5bb36a2b5a9c3b1a32105c129437f6d7311e44e071839d0afdfaba0dd8a2004 · teacher cache
sha 695023d4… (verify full sha from the wd2 receipts before citing) · wd2 student archive sha
e9c4a9ed5e6bef89d228ca877a9f9e37345e3c79dc07ba20087c218ff89fcf87 @165,387 B · refusal memo
commit 3e56d0de69 (+ scope fix 1931270e91) · rfo2 landing commit 5624ef8bdc.
SCOPE reductions allowed: analysis depth per seed. MECHANISM reductions: NONE — no verdict on
Mandate A items without reading the actual receipts; no wd3 loss weights without derivation.

## HARD CONSTRAINTS
$0 · no launches · payload law (retain anything you materialize, sha in receipts) · serializer
commits via tools/commit_autosha.sh, [no-triality] [p0-ledger-ok]; .py = 2 review passes, never
REVIEW_GATE_OVERRIDE on .py · checkpoint via tools/subagent_checkpoint.py every ~10 tool uses ·
memo .omx/research/ddm_av2_fresh_eyes_and_distillation_reopen_20260815.md with: per-seed
verdict rows {seed, verdict, evidence path, consequence}, beyond-seed findings ranked, the wd3
sealed design, and adjudication queue for MAIN. End with the vehicle frontier line
(S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600], sha 80d9c8c6…).
