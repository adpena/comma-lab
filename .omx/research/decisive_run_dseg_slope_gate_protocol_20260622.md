# Decisive-run d_seg slope gate — protocol (DEEP ANALYSIS, NO PREMATURE KILL)

**Operator directive (2026-06-22):** set the ~1-day stage-5 slope checkpoint as a decision gate, "but will
require deep analysis and reflection and no premature kill or conclusions."

This protocol binds the gate to CLAUDE.md "Forbidden premature KILL without research exhaustion" +
"Measurement-first — stop re-diagnosing" + "Recursive self-reflection protocol." The gate is an
**analysis + DEFER/CONTINUE/INVESTIGATE instrument — it has NO kill verdict.**

## When the gate fires
At **global_epoch ≈ 17,500** (~1 day at the measured ~14.2 s/ep from ep 11,643), i.e. deep into stage 5
(C1a-L7, which ends at ep 19,650) — enough of the C1a d_seg-finisher to see whether its slope steepened,
while still BEFORE the spectral Muon stage 8 (ep 24,650–29,650). Re-fires (re-analysis) at each later check.

## What the gate is NOT
It is **not** a kill switch and **not** a place to conclude "d_seg is walled / capacity-bound / dead." The
conditioning thesis PREDICTS a flat/slow d_seg through the AdamW stages (1–7): a diagonal preconditioner
cannot decorrelate the κ≈19 boundary Hessian, so d_seg descends power-law-slow there; the BULK of d_seg
closing is mathematically reserved for the SPECTRAL Muon stage 8 (O(ln 1/ε), κ-independent). Therefore **a
flat stage-5 d_seg is consistent with success, not evidence against it.** A kill/terminal verdict would
require, per CLAUDE.md: (a) research-path exhaustion (every plausible config tried), (b) exact-custody
measurement, (c) grand-council CONSENSUS, (d) documented reactivation criteria — none of which a slope
checkpoint provides.

## The measured instrument
`tools/analyze_dseg_slope_gate.py --run-dir experiments/results/yousfi_r3_taper_marginhinge_e5_20260620`
computes: per-stage d_seg endpoints + log-log slope, multi-window recent slopes (500/1000/2000 ep),
power-law fit + epochs-to-threshold projection, d_seg monotonicity (rising = a real anomaly), and the
conditioning-thesis phase flag. Its verdict taxonomy (NO KILL):
* `ON_TRACK_STEEPENING` — d_seg slope clearly negative; the finisher is biting → continue, fire branch A when it crosses.
* `ADAMW_PHASE_FLAT_AS_EXPECTED` — flat in stages 1–7; PREDICTED; continue to Muon; re-evaluate WITH the stage-8 read.
* `MEASURED_PATHOLOGY_INVESTIGATE` — a real anomaly (d_seg RISING >2%, or a measured defect); a FIX target, not a kill.
* `DEFER_PENDING_MUON` — indeterminate; defer the verdict to the spectral finisher.

## The deep-reflection checklist (the agent does this OVER the measured output — never skip)
1. **Phase check:** which stage are we in? If AdamW (1–7), flatness is expected — do NOT conclude.
2. **EMA-shadow integrity:** the exact d_seg renders the EMA shadow; confirm it's warmed (the #85 lag bug is
   fixed + the run is ep≫τ, so trustworthy) — rule out a measurement artifact before any interpretation.
3. **Loss-gradient liveness:** is the stage's loss (l7_softplus) producing a live d_seg gradient, or is it in
   a saturated/cold regime (the soft-cosine T<0.3 gradient-cliff class)? If cold/dead → MEASURED_PATHOLOGY (fix), not wall.
4. **C1a / QAT engagement:** is the C1a coder-aware reg (cat_lambda=0.01) actually active in stage 5, and is
   QAT's fake-quant noise being recovered? A mis-wired finisher is a fix, not a wall.
5. **Margin distribution:** are the residual flips shallow (training-fixable, the conditioning story) or deep
   (would indicate capacity)? Use the dseg_margin_distribution artifacts if available.
6. **Multi-window slope:** distinguish noise from trend (the 500/1000/2000-ep windows); a single flat point is not a slope.
7. **Muon-reserved expectation:** restate that the decisive read is the stage-8 Muon slope; the gate's job is to
   catch a *fixable pathology early*, NOT to pre-judge the finisher.

## Decision branches (all non-terminal)
* `ON_TRACK` / `ADAMW_PHASE_FLAT` → **CONTINUE** the burn to Muon; branch-A runbook fires when advisory d_seg
  crosses ~8.1e-4 (`decisive_run_branchA_byteclose_exacteval_runbook_20260622.md`).
* `MEASURED_PATHOLOGY` → **INVESTIGATE + FIX** the specific defect (apparatus/loss/EMA), then continue. Optionally
  launch a parallel faster-batch arm from a preserved `stage_snapshots/<stage>_end` — never kill this run.
* Genuinely stuck even after the Muon stage-8 read → that is a SEPARATE, later DEFER decision requiring the full
  no-premature-kill cascade (research exhaustion + grand-council consensus + reactivation criteria), recorded as
  `_DEFERRED_pending_<reason>_`, NOT `_killed_`.

## NO-FAKE ledger
- MEASURED: the instrument computes slopes/fits from the real trajectory; baseline @ep11.6k = ADAMW_PHASE_FLAT
  (rel_change last-2000ep −2.6%, consistent with the predicted AdamW-phase crawl).
- DISCIPLINE: no kill verdict exists in the gate; flatness ≠ wall; the decisive read is the Muon stage-8 slope.
- NOT claimed: no conclusion about d_seg's ceiling; pointer UNMOVED 0.19110.

Cross-refs: `tools/analyze_dseg_slope_gate.py` · `decisive_run_161_config_and_deepmath_optimality_audit_20260622.md`
(the conditioning thesis) · `decisive_run_branchA_byteclose_exacteval_runbook_20260622.md` (the cross-the-threshold path).
