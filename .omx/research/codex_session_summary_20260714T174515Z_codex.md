# Codex session summary: RIPO categorical trust-region warm start

**UTC:** 2026-07-14T17:45:15Z  
**lane:** `lane_ripo_fisher_isometric_trust_region_500_20260714`  
**task:** `500_ripo_fisher_isometric_trust_region_20260714`  
**research_only:** `true`  
**pointer_delta:** `0`  
**authority ceiling:** `[macOS-CPU advisory] NON-PROMOTABLE`

## Durable outcomes

- Re-derived the full five-class categorical Fisher/KL law and falsified the direct
  `sqrt(delta/p1)` logit-radius transfer.  The exact derivation and binary-reduction scope are in
  `codex_premise_falsification_ripo_multiclass_20260714_codex.md`.
- Built deterministic NumPy-fp32 clipping plus an MLX parity surface and bounded saved-array /
  fixed-head probes.  The exact-KL root clip and categorical Fisher law are locally verified;
  genuine receiver pullback remains `BLOCKED_NO_RECEIVER_PULLBACK`.
- Rebuilt the current all-int8 receiver to archive SHA-256
  `81a4c5163aa434f61489773a35862dd4b4a733219173c71e6eb8a6ef2b0613b7`, 63,664 bytes, and captured
  a fresh full-n600 sequential CPU row: 3,970,482 errors / 117,964,800 pixels,
  `d_seg=0.03365819295247396`.  Ordered pair-error vector SHA-256:
  `71aca359445b6d2e17aa62820e06e0bf19374b3094a366490cc300deaff883a2`.
- Measured full-n600 cross-space scorer tie-KL geometry.  Registered conservative delta lower
  bounds: q10 `0.002570842480640978`, q25 `0.016886937079030943`, and q50 capped by protected q01
  at `0.01847929279424884`.  These are `cross_space_output_tie_kl_budget`, not receiver/head flip
  custody.
- Own review caught and preserved two invalid candidate attempts before verdict: a camera-grid
  shape error, then use of the 117,964,800-pixel **sum** CE gradient without `/N`.  The latter fit
  was stopped at 60/600; it is a falsified instance and must never be resumed or cited as a result.

## Pending at summary creation

- Candidate v2 is being repaired to use an exact mean-loss proposal, a distinct uniform
  categorical-ratio PPO control, and realized post-reprojection/int8-deploy constraint telemetry.
- Full v2 fit/evaluation, Pose, archive parse-back, contest CPU/CUDA rows, V9 DSL wiring, and any
  pointer movement remain owed.

## Canonical next step

Finish the source-bound v2 repair and focused tests, run a four-pair real integration smoke, then
advance the resumable full-n600 fit/evaluation only if mean-loss scaling, baseline replay, and
post-deploy constraint custody all pass.  Exact-KL is primary; local-Fisher arms are approximation
probes and must report exact/local mismatch.  Confidence-band stability is not RIPO Proposition
4.1 homoscedasticity.

## Inbox cursor

No directive newer than `2026-07-14T17:00:15Z` had been consumed when this summary was created.

