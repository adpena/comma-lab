# Capstone synthesis: two independent subagents CONVERGE on base_ch=20; the gate is int8-REALITY, not a long train (2026-06-11)

**Authority:** `[macOS advisory]`. Pointer UNMOVED at 0.19109982 [contest-CPU]. This memo synthesizes two
independent subagent deliverables landed this session + corrects an over-claim I made.

## The two subagents (independent, distinct axes) CONVERGED on the central call
1. **d_seg-crux + param-Pareto** (`dseg_crux_objective_and_param_pareto_20260611.md`, commit `1949305e1`):
   the measured param↔d_seg curve is a **FLAT BASIN** (~5.6e-4 across 85K–180K params) while decoder bytes
   scale ~linearly → the **smallest basin-floor base_ch wins on rate**. base_ch=20 (85K, stored_latent
   ≈101 KB, rate 0.067) → predicted S ≈ **0.140 (sub-0.15)**. The d_seg lever is the EXISTING PR95 8-stage
   seg-loss-FORM SCHEDULE (ce→softplus→smooth→l7) + epochs, NOT a new loss: a boundary-weighted TCKD/DKD
   objective was MEASURED-LOST 3.07× to KL-T2 (`ab_boundary_tckd_vs_kl_t2_20260531.json`), and CE already
   puts 99.4% of its gradient on the boundary band. **Reuse the schedule; don't reinvent the loss.**
2. **Adversarial review** (`adversarial_review_post_ema_fix_picture_20260611.md`): independently found the
   "need 162–229K frontier-class params" thesis is FALSE — it rested on a 2026-05-09 council *projection*
   AND on reading the frozen EMA shadow (the bug fixed in `f771e6e00`). The live daemon shows stage 2
   (softplus) **starting at 0.0165 and descending** — the curriculum IS breaking the CE plateau.

**Convergent verdict: base_ch=20 (85K) is the right capstone size — NOT frontier-class.** Two methods,
same answer. This REVERSES `capstone_adversarial_synthesis_..._20260611T015018Z.md`'s "frontier-CLASS
params" correction (which itself read the poisoned shadow). The small-learned-basis thesis — the operator's
instinct — is vindicated on BOTH the d_seg axis (EMA fix) AND the param/rate axis (flat basin).

## HONEST CORRECTION (my over-claim, retracted)
I said "pose is SOLVED via the stored_latent carrier." **Retracted to: pose is held by a SEPARATE stored
6-dim GT pose + FiLM (Quantizr store-the-answer trick), NOT the 28-d latent** (`export.py:13`,
`vq_nerv_bundle.py:228`). And it is measured on **48 pairs, LIVE float render, the proto scorer** — never
on int8, never at 600 pairs, never on the real FastViT PoseNet. d_pose is a per-pair MEAN (could be
few-pair-dominated under the concave √). "Pose held cheaply" is plausible (Quantizr shipped it at 0.33)
but UNVERIFIED at the contest operating point. Re-verify at int8/600-pair/real-PoseNet before any claim.

## THE GATE (adversarial review's #1 lever): int8-REALITY before any long train
Everything decisive rests on numbers that must be made REAL:
- **A2 (reloaded-int8) IS closed in the runner** — verified `run_capstone_campaign.py:292`
  `score_reloaded_int8_archive`; the advisory score uses the RELOADED int8 d_seg/d_pose (+ live-float gap).
  BUT the STREAMING telemetry (`trajectory.jsonl`) is LIVE FLOAT — the int8 number lands only in the final
  `capstone_result.json`. So the mid-run d_seg≈0.02 / d_pose≈1e-3 I quoted are float, not the archive.
- **A3 (bicubic numpy-inflate) is the remaining gap.** `score_reloaded_int8_archive` re-scores through the
  BRIDGE (bilinear resize); the ACTUAL contest decode is the numpy `inflate.py` (bicubic camera upscale).
  The TRUE predictor = numpy `inflate.py` on the int8 archive → bridge score. This must be run on the
  daemon's emitted `archive.zip` and compared to both the live-float and the reloaded-int8-via-bridge.

## NEXT-BUILD SEQUENCING (changed by the synthesis — do NOT launch a long/frontier-class train now)
1. **Daemon finishes** (`capstone_c1prime_honest_b20_n48`, marker-on-exit) → emits `archive.zip` +
   `capstone_result.json` with the RELOADED-int8 advisory S (A2). This is the first honest int8 number.
2. **Run the A3 check**: numpy `inflate.py` (bicubic) on that `archive.zip` → bridge score → the TRUE
   advisory S. Compare float vs reloaded-int8-bridge vs numpy-inflate-bicubic (the 3-way gap IS a finding).
3. **IF the int8/bicubic advisory S survives < ~0.19** AND d_pose holds → byte-close (add the ~30-LOC
   inflate.sh wrapper) → paired contest CPU+CUDA exact eval (the pointer-mover). base_ch=20, NOT frontier.
4. **IF int8 degrades d_seg/d_pose** → that's the real crux; fix the export/quant path (QAT stage, the
   curriculum already has it) before scaling. Do NOT spend a 600-pair train on a float mirage.
5. The 600-pair run is the EVENTUAL gate but is CPU-scorer-bound (~18 min/epoch); the 48-pair curriculum
   is the affordable proxy that resolves the curriculum-breaks-plateau question first.

## Banked / in-flight
- HiNeRV bilinear-skip + grid-PE upgrade: building (opt-in, default-off, parity-tested) — a BYTE/BD-rate
  lever useful regardless of param count; readies a stronger candidate.
- Lever B: banked negative (−59% rate headroom + measured geometry; pose-blind palette can't carry pose).
- torch `tac.training.EMA` warmup: pending training.py CRITICAL review debt (task #86; dormant surface).

## Burning question (unchanged, sharpened)
Does the base_ch=20 curriculum's int8+bicubic archive advisory S land < 0.19 (and ideally < 0.15) at the
real operating point — or does int8 quant / the 600-pair scale-up / the real PoseNet break the float
advisory? The daemon's `capstone_result.json` + the A3 check answer the first half; 600-pair answers the rest.
