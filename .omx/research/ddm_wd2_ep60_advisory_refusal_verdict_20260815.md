# wd2 ep60 advisory n600 verdict: REFUSED — the sub-0.15 composed route dies at this form (MAIN, 2026-08-15)

## The measurement [macOS-CPU advisory via mirror d5bb36a2, n600, attempt 2]
Receipt: .../advisory_n600_cpu/contest_auth_eval.json (durable, work dir retained).
- avg_segnet_dist **0.00117677** vs base 0.00029611 → **Δd_seg +8.81e-4 = 8.2× OVER the
  ≤1.07e-4 admission bar**. Seg cost alone +0.0881 S vs a rate prize of −0.0116 S.
- avg_posenet_dist **0.09198625** vs the CPU-axis base expectation ~1.45e-4 (21× CUDA
  degradation per #1054) → ~634× the axis-degraded base. The student's rendered frames
  destroy the pose signal (~0.21 S CUDA-equivalent pose term).
- Canonical S on this axis 1.186896 (advisory; env-mismatch stamped by the harness — the
  DELTA vs base is the decision quantity; seg is CPU-stable, and the pose gap is far beyond
  axis noise).

## Verdict + scope
**REFUSED.** The composed ≈0.1480 projection via the ep60 flattened d4/w64 student is DEAD.
verdict_scope: instance — the ep60 flattened d4/w64 decode-MSE-only student on the hv1 base;
the nested-width FAMILY stays open (parked with its measured trajectory, reactivation =
scorer-aware loss or larger width per the routing below). The
FAMILY question (can any narrow student hold scorer fidelity?) stays open but is now priced:
- Trajectory (TRAIN_RESULT.json): decode_mse_uint8 1,582 (ep1) → 50.7 (ep60), RMS ≈ 7 uint8
  levels/px. Train loss log-slope DECELERATING (−0.0210/ep at ep20→40 → −0.0128/ep at
  ep40→60). Closing the seg gap alone at the current rate ≈ 160+ epochs pre-deceleration;
  the pose gap (~600×) is beyond any continuation of this config. Training completed its
  planned 60 ep (no later checkpoints); ~8–10 h Metal for a bad bet → NOT continued.
- Mechanism: the loss was teacher-decode MSE only. PoseNet needs near-photometric exactness;
  a w64 student at RMS 7/255 noise cannot hold it. Any successor needs either far more
  capacity (eroding the rate prize) or a scorer-aware loss — that is the js1 joint line's
  territory, not a cheap distillation.

## Routing (rfo2's measured route order CONFIRMED by this refusal)
Route order stands: **mixed precision → carrier rank/refit (22,032 B pool) → nested-width
(now parked with its measured curve) → token drop/coder refit.**
1. FIRED: codex arm ddm_mp2 (the #1063 reserved-next) — receiver-close mz2's retained
   q3/q4 (−823 B) + 6 FiLM-row sparsity candidates (−130..−2,051 B), advisory n600 each
   through the mirror chain (admit < −3.5e-6 net), then the carrier rank/refit build vs the
   22,032 B basis+coeff pool.
2. wd2 assets retained (payload law): ep60 checkpoints + teacher cache (1.83 GB, sha
   695023d4…) + both advisory work dirs. The teacher cache serves any future student arm.
   **AMENDMENT (operator 2026-08-15 "Also perhaps distillation is not dead"): the family is
   elevated from parked to ACTIVELY REOPENED — ddm_av2 Mandate B owns the wd3 successor
   design (scorer-aware distillation loss through R, derived weights, in-loop realized
   verdicts, dense-w56/factorized/wider arms w/ rate-erosion table), gated on the
   same-instrument hv1-base advisory row now in flight.**
3. Metal slot: free after r5 self-terminates; no heavy launch until a candidate demands it.

## Ops notes
- Attempt 1 died to `._0.mkv` mirror contamination (AppleDouble on ExFAT) — the #812
  denominator gate refused CORRECTLY; cure + structural fix routed in #1064. Attempt 2 ran
  clean end-to-end through the new mirror + python-shim chain: the LOCAL advisory chain is
  now proven working at n600 (~35 min wall).
- Both watcher ALERT classes this evening adjudicated: one real crash (cured), one rc=0
  false-positive (#1064).

Vehicle frontier UNCHANGED: **S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**
(below public #1) · Modal ≈ $6.3/$20.
