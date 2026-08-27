# ddm_qbt1_qbflow_trainer_build — build + review the real scorer-in-loop QBFLOW trainer against the frozen QBF1 ABI (the sealed fire order's one missing precondition); NO launch from the arm

## MANDATE

Operator standing GO (08-21 "whatever it takes... frontier score lowering"). QBFLOW's
rate gate CLEARED (aa58109f86 + c6dee964cb: exact initialized archive 107,582 B sha
0c833881…, 30,404 B under the 137,986 B cap; trained-entropy projection 122,797 B).
The SEALED_TRAINING_FIRE_ORDER
(`/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/SEALED_TRAINING_FIRE_ORDER.json`,
schema ddm_qbflow_training_fire_order.v1) names its ONE missing precondition: "MAIN...
lands/reviews a real QBFLOW scorer-in-loop trainer consuming this exact packet ABI."
This arm builds that trainer to review-complete, launch-ready state. MAIN fires the
training stages on the governed Metal slot afterward — NO launch, NO Metal, NO Modal,
NO full-n600 scorer job from the arm.

## SCOPE

1. **Recall-first (m122)**: the fire order's hard gates BIND the design verbatim —
   chunk_pairs ≤30 · memory peak ≤116 GiB (124,554,051,584 B) · trained checkpoint MUST
   re-encode through the real coder (initialized rate does NOT transfer) · same-budget
   QBW1 control required · admission = complete ≤137,986 B + d_pose_hat ≤1.25e-4 +
   S_hat <0.12 (no2 §5 @ d0fe0168b5). Consume: the frozen QBF1 ABI + receiver + runner
   (@ aa58109f86 — the trainer consumes THIS packet ABI exactly; any ABI change is a
   REFUSAL, not an edit) · the qbflow verdict memo (@ c6dee964cb — the staged plan:
   stage_03 joint boundary/interior birth → stage_04 sensitivity/precision + re-encode →
   stage_05 gate) · the WD3 trainer as the family REFERENCE FORM for scorer-in-loop
   mechanics (`experiments/ddm_wd3_sealed_3d9e021d07_runner.py` — its chunked
   materialization, resume identity, controller binding; the r1/r2 lesson: one 60-pair
   full-autograd chunk breaches the MPS watermark, chunk ≤30 from birth) · the aligned
   expected-flip-margin seg law (w96b, seed-robust 2.03× on its own axis at BOTH seeds —
   `ddm_w96b_seed20260816_aligned_verdict_and_family_closure_20260827.md` @ 594dfe3510).
2. **Build the trainer** (`experiments/ddm_qbt1_qbflow_trainer.py` + tests):
   - Objective per the fire order's stage_03: joint realized-through-R Seg
     interface/RGB descent + pose6 descent from step zero; NO fixed paint, NO post-hoc
     pose. Seg law = the vindicated expected-flip-margin form (cite the w96b receipts;
     do not regress to CE-by-default — the derived-schedule laws bind).
   - Realization in-loop: render → R (bicubic/bilinear per upstream semantics) → uint8
     STE → frozen scorers; eval_roundtrip is NON-NEGOTIABLE; scorer preprocess
     differentiable (the yuv6/no_grad trap).
   - EMA per the non-negotiable: decay resolved through the run-geometry LawRef
     (ema_decay_run_geometry_v1), shadow saved as the inference state, snapshot+restore
     at eval only.
   - Resumability P0: per-stage distinct atomic checkpoints + periodic intra-stage
     saves + resume-from-disk identity test; chunk_pairs ≤30 hard-coded ceiling with a
     config assert (not a default).
   - Stage_04 hooks: per-role receiver/scorer sensitivity measurement + real precision
     options + MANDATORY re-encode of every trained checkpoint through the real coder
     into the complete framing (the rate gate is re-proven per checkpoint, never
     assumed from the initialized packet).
   - Stage_05: the no2 §5 gate evaluator + the same-budget QBW1 control leg, emitting
     RESULT.json in the qbflow observability schema.
3. **Verification without launch**: focused tests + a bounded n≤4-pair CPU smoke that
   proves the full loop (encode→train-steps→re-encode→gate arithmetic) RUNS and
   resumes bit-faithfully — mechanism verification only, clearly labeled, never a
   verdict. Memory projection at the real n32 config (the #205 lesson: project the
   MATERIALIZATION path, not just the step loop) written into the launch request.
4. **Emit the compiled launch request** for MAIN: config + measured smoke receipts +
   memory projection vs the 116 GiB ceiling + schedule estimate + the fire-order gate
   checklist pre-filled. MAIN reviews the .py (2 visible passes verified), claims
   lanes, and fires on the free Metal slot.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO training launch/Metal/Modal/full-n600 scorer from the arm.
  Custody root `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/` (extend;
  live df at write time). ALWAYS KEEP THE PAYLOAD (every smoke checkpoint + re-encoded
  packet retained).
- Serializer commits w/ post-edit `--expected-content-sha256`; ALL `.py` = 2 genuine
  visible review passes (this is the review the fire trigger demands — it is the
  deliverable, not overhead).
- The frozen QBF1 ABI is IMMUTABLE input: the trainer adapts to it; if the ABI cannot
  support a needed gradient path, that is a typed REFUSAL routed back to MAIN, never a
  silent schema edit.
- Axis honesty: smoke rows are mechanism receipts, score_claim=false everywhere.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends — the distortion wall this trainer faces)

- Every trained carrier died on distortion at its first fit: born-small 66–209× · nr1
  349× · W72 46× · W96-aligned pose 185–204× gb1. The QBFLOW escapes are named (joint
  birth, comb bracketing lane-dash ~25 cycles, receiver-derived Road conditioning) —
  the trainer must IMPLEMENT all three, or the escape claim is fake.
- Fixed-high-β hosc: measured saturation/divergence — closed; use the stable
  trainable-slope/annealed forms per the witness-line receipts.
- The #205 OOM chain + r1/r2: full-batch scorer forwards at scale breach memory —
  chunked verdicts + chunk≤30 materialization are structural, not tunable.
- qbw1/qbw2: no serialized boundary payload may creep back in (the generate-don't-
  serialize law, memory explicit-boundary-floor-equals-gb1-archive-generate-dont-serialize).

## OPTIMAL FORM

- Reference form + provenance pins: WD3 sealed runner (scorer-in-loop mechanics
  exemplar) · QBF1 ABI + receiver (@ aa58109f86) · qbflow verdict (@ c6dee964cb) ·
  fire order sha-bound at consumption · no2 §5 (@ d0fe0168b5) · w96b seg-law receipts
  (@ 594dfe3510) · gb1 sha ba1f3830…88a3e4 (control object). SCOPE reductions (legal):
  n≤4 CPU smoke; n32 as the first governed window. MECHANISM reductions FORBIDDEN:
  real R+uint8+frozen-scorer loop in the smoke (no proxy losses standing in); real
  coder re-encode; the three named escapes implemented, not stubbed.
- **PRIOR-LAW PREDICTION (falsifiable):** the build completes and the smoke proves the
  loop; the open empirical question this trainer will settle at MAIN's fire is whether
  joint-from-birth escapes the pose wall — the family record says first-fit distortion
  fails; the object change (sy2) says this is the first family where pose shapes the
  interiors from step zero. No number is predicted for the trained rung — that is what
  the burn measures. FALSIFIER for THIS arm: an ABI gradient-path refusal or a smoke
  that cannot resume bit-faithfully → typed blocker back to MAIN.

## DELIVERABLE

`.omx/research/ddm_qbt1_qbflow_trainer_build_20260827.md` — trainer inventory (files,
tests, 2-pass review receipts) + bounded-smoke receipts (loop + resume identity) +
memory projection + the compiled launch request w/ pre-filled fire-order checklist,
OR the typed ABI refusal. Commit via the serializer. End with the own-vehicle frontier
line (gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]).
