---
title: "Yousfi-order round-3 stage-1 read (taper + margin_hinge + E#5, full-MPS pose path) — TWO prior findings REVERSED: (1) margin_hinge HOLDS pose on the working path; (2) the dseg-aware byte-neutral taper is at PARITY-to-marginally-AHEAD of vendored at matched epochs, NOT 'weak'. Stage-2 transition (E#5 decisive test) ~1.5h out."
authority: "[contest-CPU advisory / MLX-trained] — pointer UNMOVED 0.19110; $0; NO paid; NO score claim"
score_claim: false
promotion_eligible: false
pointer_moved: false
date: 2026-06-21
verdict: MARGIN_HINGE_HOLDS_POSE_CONFIRMED · TAPER_AT_PARITY_TO_MARGINALLY_AHEAD_NOT_WEAK · STAGE2_E5_TEST_PENDING
cross_refs:
  - .omx/research/yousfi_r2_step1_gate_marginhinge_validated_pose_path_broken_20260620.md
  - .omx/research/RECURSIVE_REVIEW_pr95_math_optimization_synthesis_20260619T231500Z.md
  - experiments/results/yousfi_r3_taper_marginhinge_e5_20260620/   # this run (pid 29988)
  - experiments/results/yousfi_r2_marginhinge_fullmps_20260620/    # GREEN vendored baseline / KD teacher
---

# Round-3 stage-1 read — the decisive measured A/B (taper vs vendored, both with margin_hinge on the working pose path)

Run `yousfi_r3_taper_marginhinge_e5_20260620` (pid 29988, ~9.6 h, ep ~2535, stage 1 of 8). Full-MPS gradient,
CPU-authority async eval. `[contest-CPU advisory]`; pointer UNMOVED 0.19110; $0; no paid; no score claim.
Config: `--no-split-by-head --seg-margin-hinge --stage-lr-warmup-frac 0.03 --taper-channels 16,16,17,19,19,14,10
--kd-warm-start-dir <GREEN/best> --no-muon-lr-floor-fix`.

## Reversal 1 — margin_hinge HOLDS pose on the full-MPS path (the open question from r2 is answered YES)
The r2 step-1 gate left this INFERRED: does margin_hinge hold pose on the working `--no-split-by-head` path, or
does the seg-crank trade pose away even when the pose gradient is live? **Measured: it HOLDS.**

| ep | d_seg | d_pose | S |
|---|---|---|---|
| 25 | 0.01669 | 0.03488 | 2.316 |
| 325 (post KD-warm settle) | 0.00357 | 0.00102 | 0.516 |
| 1000 | 0.00279 | 0.00062 | 0.416 |
| 1600 | 0.00249 | 0.00047 | 0.376 |
| 2350 (best) | **0.00233** | **0.00031** | **0.3477** |

d_pose descended monotonically to 3.1e-4 and is still falling — the opposite of the r2 bind-all arm (d_pose 0.786
at ep25, never recovered). Confirms: the r2 pose break was the **split-by-head + equimarginal** path, NOT
margin_hinge. The corrected wiring (margin_hinge on the pose-training full-MPS path) is sound.

## Reversal 2 — the dseg-aware byte-neutral taper is NOT weak; it is at parity-to-marginally-ahead
Earlier reads called the taper "weak" (#121 DOWNGRADED). That was an apples-to-**oranges** comparison: taper at
ep~925 (d_seg 0.00283) vs GREEN's *converged* number (~0.0023 at ep4525). The matched-**epoch** read reverses it:

| metric @ ep~2400 | GREEN vendored [20,20,20,15,11,10,10] | TAPER dseg-aware [16,16,17,19,19,14,10] | Δ |
|---|---|---|---|
| d_seg | 0.00240 | **0.00233** | −2.9% (taper better) |
| d_pose | 0.00026 | 0.00031 | GREEN tighter (both ~3e-4, excellent) |
| S | 0.3508 | **0.3477** | taper −0.0031 |

Byte-neutral verified: dseg-aware = 83,422 params vs vendored 83,356 = **+66 (+0.079%)**. The reallocation shifts
capacity from coarse early stages (16 vs 20 at stages 0–2) into the mid-resolution bands (19,19 vs 15,11 at
stages 3–4) where SegNet decision-boundary flips concentrate (the 192×256 band per the small-basis micro-macro
audit). The −2.9% matched-epoch d_seg edge is the first measured signal that "put params where the flips are"
helps. **Honest caveat:** the edge is small (~3%, advisory, could be noise); GREEN's best was 0.00224 (ep2950),
which the taper hasn't reached yet (ep2535). The decisive test is whether stages 2–8 widen or close the gap. The
operative verdict is at-parity-and-byte-neutral, so it cannot HURT — and shows a small positive.

## Deep-math review (standing directive)
1. **Stage-1 d_seg power law** (28 evaluated pts, ep≥400): `d_seg = 0.0124 · ep^(−0.217)`. Extrapolated epochs to
   the sub-0.15 d_seg target (0.000322) **in stage-1 CE geometry = 2.07×10⁷** → INFEASIBLE in CE alone. Stage 1's
   job is to seat a good basin (~0.0023 d_seg, done/plateauing); the d_seg-finishing stages (2–8: softplus τ →
   smooth → QAT → C1a-L7 → λ/σ sweeps → Muon) must break the plateau via their sharper/quantized/orthogonalized
   geometry. The power-law extrapolation only binds WITHIN one loss geometry — it is the formal reason the
   8-stage curriculum exists.
2. **Marginal-value law at the best point** (d_seg 0.00233, d_pose 0.00031): ∂S/∂d_seg = 100 (const);
   ∂S/∂d_pose = 5/√(10·d_pose) = **89.4** ≈ 0.89× seg; we sit at 1.25× the pose crossover (d_pose=2.5e-4 where the
   two marginals equalize). Near-equal *marginals*, BUT the *removable headroom* is d_seg-dominated: pushing d_seg
   to its sub-0.15 target removes 0.201 of S; pushing d_pose to 0 removes only 0.056. **d_seg holds 78% of the
   removable S → it remains the binding axis** even though the per-unit sensitivities are now comparable.
3. **S decomposition**: seg_term 100·d_seg = 0.2335 (67% of S); pose_term √(10·d_pose) = 0.0559; bc20 small-basis
   rate floor ≈ 0.0594. Projected byte-closed S ≈ 0.3488 ≈ reported 0.3477 (rate already counted).

## Near-term decisive milestone (~1.5 h out): the E#5 stage-2 transition test
Stage 1 ends at ep3000 (taper at ~2535, ~465 ep / ~1.7 h away). GREEN (NO E#5 warmup) kicked d_pose **3×** at its
stage-2 boundary (0.00021 best → 0.00062 at ep4525, stage2_softplus). The taper run carries
`--stage-lr-warmup-frac 0.03`, so the boundary is the decisive E#5 validation: does the per-stage LR warmup-after-
restart damp the trunk-slam pose-kick? Concrete pass/fail at the next read.

## NO-FAKE ledger
- MEASURED: full d_seg/d_pose/S trajectory (CPU-authority); taper −2.9% d_seg vs GREEN at matched epoch; byte-
  neutral +66 params; power-law b=−0.217.
- INFERRED (not yet measured): that stages 2–8 break the stage-1 d_seg plateau (the whole sub-0.15 bet);
  that the E#5 warmup damps the stage-2 pose-kick (decided at the next read).
- NOT claimed: no score moved; pointer UNMOVED 0.19110; S≈0.348 advisory is far above frontier 0.191 because
  the run is still in stage 1 — the d_seg-finishing stages have not run.

## Observability surface
`yousfi_r3_taper_marginhinge_e5_20260620/torch_vehicle_trajectory.jsonl` (evaluated=true rows carry CPU-authority
d_seg/d_pose/S). GREEN: `yousfi_r2_marginhinge_fullmps_20260620/`. Axis `[contest-CPU advisory]`,
score_claim=false, pointer 0.19110.
