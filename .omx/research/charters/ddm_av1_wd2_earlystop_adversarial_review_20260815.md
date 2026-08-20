# ddm_av1_wd2_earlystop_adversarial_review — fresh-eyes adversarial pass on the live composition chain

## MANDATE (operator 2026-08-15: "No naive or toy" + "May need adversarial review")
Round-1 ADVERSARIAL review (finding round; the 3-clean-pass seal cycle continues at the
compose/freeze boundary) over the 08-15 chain: the wd2 width-distillation build + LIVE config,
the e960 governed early-stop + ep634 checkpoint selection, and the hv1 compose-chain assumptions.
Your job is to REFUTE, not to confirm. Every finding: severity + concrete failure scenario +
receipt (file:line / sha / arithmetic). Every seeded hypothesis gets an explicit
CONFIRMED / REFUTED / INDETERMINATE-with-named-resolving-measurement verdict — no silent skips.

## REVIEW TARGETS (pinned)
- wd2 build commit 706a8f9d9680989aa5d0c1ff67d2950ffa88df12: experiments/ddm_wd2_width_distillation_build.py
  (sha e1a0a0650b929d31b749ac2734d12dd52b0c8ba7786cdbde89b1206029db1c90) +
  experiments/ddm_wd2_student_receiver.py (9024b563a4895a695927bda9f8f2f6aaf6adc2780f074eebb6760610c3bebfa0) +
  .omx/research/ddm_wd2_width_distillation_build_20260815/launch_ticket.json + memo.
- Teacher-cache receipt: /Volumes/APDataStore/pact/ddm_wd2_width_distillation/teacher_cache_e480b/TEACHER_CACHE_RESULT.json
  (payload sha 695023d4ca56e14f53f1e90b56134821c3c0a0c66f9b07f6aa6bd6ffdf9f4ebd, 1,831,204,800 B).
- Selector: tools/select_hpac_checkpoint.py (landed 5624ef8bdc) + its receipt
  /Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full_e480b_e960/endpoint_closure/checkpoint_selection.json
  (ep634, joint 130,393 B, top1 0.0018945397271050348) + governed_early_stop_receipt.json + the fit
  receipt /Volumes/VertigoDataTier/pact/ddm_lh2_20260815/midrun_descent_fit_ep568.json.
- hv1 charter .omx/research/charters/ddm_hv1_harvest_compose_ep508_20260815.md + the ep634 retarget
  (retained /Volumes/VertigoDataTier/pact/ddm_hv1_harvest_compose/retained/epoch_0634.pt,
  sha 5007beae7af7789758092f12f49096e13692e2e59850c85eb4642cd6fad147ec).

## SEEDED HYPOTHESES (attack these; add your own beyond-seed sweep)
H1 SELECTOR PROXY VALIDITY: select_hpac_checkpoint ranks by (25/37545489)·joint_bytes + 100·top1_error,
   where top1_error is TOKEN-level teacher/GT mismatch — NOT d_seg through the scorer. Could the argmin
   ranking INVERT under the real n600 scorer between ep508 and ep634 (or vs neighbors)? What monotonicity
   assumption is being made, is it stated anywhere, and what is the cheapest falsifying measurement?
H2 TEACHER-CACHE 84s: verify from code + receipts that the cache is the decode output of the FROZEN
   teacher through the REAL receiver path at full geometry — that the pair-0 byte-identity gate
   (builder ~:492-503) executed on THIS invocation and the loop wrote all 600 pairs (no resume shortcut,
   no cached reuse). 84 s for 600 MPS decodes = plausible? Show the arithmetic.
H3 EMA DECAY: config claims "decay derived as 1 - 2 / total optimizer updates." Check against the
   registered LawRef ema_decay_run_geometry_v1 (p0_ema_calibration). Derived, or an invented formula
   wearing a derivation's clothes? If divergent, quantify the effect at 60 epochs.
H4 STE SATURATION: the train forward clamps to [0,255] BEFORE round (builder ~:863-871). Gradient
   through clamp is zero outside range — saturated pixels are gradient-dead. Known-accepted property
   or a live training hazard for a student matching a uint8 teacher (how much teacher mass sits AT 0/255)?
H5 PACKET ARITHMETIC: "19,465 B raw packet" vs the 19,606 B rate-only ceiling vs the 34,763 B frozen
   semantic pool — same-coder honest? Is the student packet compared RAW while the frozen section is
   post-brotli (or vice versa)? The admission falsifier (beat 2,051 B structural alternative) — is
   2,051 B measured on the same axis?
H6 TWO-INSTRUMENTS-CONVERGED: MAIN claimed the fit's "≤~480 B remaining" matching realized 482 B
   (ep508→ep634) is two independent instruments converging. Verify both numbers from receipts; are the
   instruments actually independent (fit consumed the same log the selector read)? Numerology check.
H7 EP634 BYTE-CLOSE READINESS: does epoch_0634.pt carry EVERYTHING the rx2/mc36 byte-close needs
   (EMA shadow vs live weights — which does the export chain consume? cfg keys? per-stage state)?
   A checkpoint that byte-closes to a DIFFERENT coder state than the selector's joint-bytes estimate
   would break the selection premise.
H8 ACCUM GEOMETRY: batch_size=1 × accumulation_steps=8 claimed to "preserve the donor's effective
   batch geometry." Check the donor's actual geometry (tools/train_ddm_cl1_hpac_capacity_mps.py,
   conventions donor). Preserved, or asserted?

## HARD CONSTRAINTS
- LIVE PROCESSES SACRED, READ-ONLY: wd2 training pid 28814 (+ watchers) and the hv1 Opus agent's
  stores. NO Modal, NO scorer runs, $0. You may read any receipt/payload metadata.
- No code edits except trivial-and-tested doc/comment fixes; route all fixes to MAIN as findings.
- Serializer commits (tools/commit_autosha.sh), [no-triality] [p0-ledger-ok]. Memo:
  .omx/research/ddm_av1_wd2_earlystop_adversarial_review_20260815.md with the findings table,
  per-hypothesis verdicts, DEAD-ENDS + LIVE-HYPOTHESES, and a ranked MAIN-adjudication queue.
  Checkpoint via tools/subagent_checkpoint.py every ~10 tool uses. End with the vehicle frontier line.

## OPTIMAL FORM
- Family reference: the recursive adversarial review protocol at its landed form (CLAUDE.md; verify
  by RE-DERIVING from primary artifacts, never trusting memo prose; assumption-challenge axis
  mandatory: state the shared assumption this chain operates within and whether violating it
  changes the verdict). MECHANISM reductions: NONE — a prose-only review that does not re-derive
  numbers from receipts is inadmissible. SCOPE: the 08-15 chain listed above, one finding round.
- Provenance pins: wd2 706a8f9d96 · selector 5624ef8bdc · early-stop receipt schema
  governed_early_stop_receipt.v1 · e480b archive sha e3e6f440b45bbb92… · frontier
  S 0.1600920261571558 @ 183,502 B [contest-CUDA T4 n600].
