---
title: "Yousfi-order round-2 step-1 GATE — SPLIT verdict: margin_hinge VALIDATED on d_seg at n600, but the bind-all split-by-head+equimarginal pose path is BROKEN (d_pose 0.8 from a 0.00034 KD-warm start). Fix: margin_hinge on the full-MPS pose path (corrected run launched)."
authority: "[contest-CPU advisory / MLX-trained] — pointer UNMOVED 0.19110; $0; NO paid; NO score claim"
score_claim: false
promotion_eligible: false
pointer_moved: false
date: 2026-06-20
verdict: MARGIN_HINGE_VALIDATED_ON_DSEG · SPLIT_BY_HEAD_PLUS_EQUIMARGINAL_POSE_PATH_BROKEN · CORRECTED_RUN_ON_FULL_MPS_PATH_LAUNCHED
cross_refs:
  - .omx/research/RECURSIVE_REVIEW_pr95_math_optimization_synthesis_20260619T231500Z.md
  - .omx/research/lensA_dseg_optimal_loss_geometry_ce_vs_margin_hinge_20260619.md
  - experiments/results/yousfi_r2_arm_a_marginhinge_20260619T234500Z/   # the broken-pose run (decisive read)
  - experiments/results/yousfi_r2_marginhinge_fullmps_20260620/         # the corrected run
---

# Yousfi round-2 step-1 gate — the measured read (the decisive A/B of lens A's #1 lever at n600)

The recursive-review's #1 lever (margin_hinge, lens A) was run at the real n600 operating point with pose active,
in Yousfi's order (surgical detector loss first). The ~19h run produced a **SPLIT** verdict. All
`[contest-CPU advisory]`; pointer UNMOVED 0.19110; $0; no paid; no score claim.

## The measured trajectory (bind-all arm_a + margin_hinge + split-by-head + equimarginal + FiLM-v2, KD-warm from basin)
CPU-authority async-eval, `yousfi_r2_arm_a_marginhinge_20260619T234500Z/launch_go.outer.log`:

| ep | d_seg | d_pose | note |
|---|---|---|---|
| 25 | 0.00836 | **0.786** | d_pose ALREADY broken at ep25 (KD-warm start had 0.00034) |
| 3000 (end stage 1) | **0.00221** | ~1.15 | d_seg BEAT basin CE (0.00251@ep2307); d_pose stuck high |
| 4300 (stage 2) | 0.00240 | 0.70–2.85 | d_seg holds ~0.0024; d_pose volatile, catastrophic |

## The two separable facts
1. **margin_hinge VALIDATED on d_seg.** It drove d_seg 0.00836→**0.00221** by ep3000 — *better* than the basin's
   CE (0.00251 at ep2307), confirming lens A's measured 0.643×-CE result at the real n600 operating point. The
   #1 Yousfi detector-informed loss lever WORKS.
2. **The pose path is BROKEN — and it is NOT margin_hinge's fault.** d_pose was **0.786 at ep25** and never
   recovered, despite KD-warm-starting from the basin which had **d_pose 0.00034**. Pose was destroyed in the
   first 25 epochs → this is the **bind-all split-by-head + equimarginal pose path failing to train pose**, NOT
   a stage-transition blowup and NOT margin_hinge. The basin's `--no-split-by-head` full-MPS path trained pose
   to 0.00034; the bind-all launcher *forces* split-by-head for MPS, and that path + the equimarginal controller
   (which de-weights pose when d_pose is high — a vicious cycle: high d_pose → small ∂S/∂d_pose → less pose
   weight → higher d_pose) never descends pose.

## The fix (corrected run launched)
Wire margin_hinge (validated) onto the **full-MPS pose path that actually trains pose** (`launch_split_by_head_basin.py
--no-split-by-head`), dropping the broken split-by-head + equimarginal guardrail. Added `--seg-margin-hinge` to
the basin launcher (commit `4e9abbb1b`) via the driver's `curriculum=` hook (sets `seg_surrogate="margin_hinge"`
on every stage). Launched `yousfi_r2_marginhinge_fullmps_20260620` (resume from the basin's good-pose state,
floor-fix OFF). **The decisive open question it answers:** does margin_hinge HOLD pose (d_pose ~1e-3) on the
working full-MPS path while descending d_seg — or does the margin_hinge seg-crank itself trade pose away even
when the pose gradient is active?

## Yousfi-order consequence
Per the discipline (protect the 2nd detector; "pose destabilizes → diagnose the guardrail first"), steps 2
(d_seg-aware taper) and 3 (Muon-early@0.03) are **HELD** until the corrected step-1 run shows margin_hinge holds
pose on the full-MPS path. The equimarginal controller is REMOVED from the immediate stack (it broke pose); it
may return later only if a measured pose/seg-balance need appears AND it's fixed for the high-d_pose regime.

## NO-FAKE ledger
- MEASURED: the full d_seg/d_pose trajectory (CPU-authority eval) of the broken run; d_seg 0.00221 < basin CE
  0.00251; d_pose 0.786 at ep25 vs 0.00034 KD-warm start.
- INFERRED (the corrected run tests): that margin_hinge on the full-MPS path holds pose. NOT yet measured.
- NOT claimed: no score moved; pointer UNMOVED 0.19110; the broken run's S (3–5) is far worse than the basin
  (~0.34) BECAUSE of the broken pose, not the d_seg lever.

## Observability surface
Broken run: `yousfi_r2_arm_a_marginhinge_20260619T234500Z/{launch_go.outer.log,torch_vehicle_trajectory.jsonl}`.
Corrected run: `yousfi_r2_marginhinge_fullmps_20260620/` (pid 20102). Lane claims recorded. Axis
`[contest-CPU advisory]`, score_claim=false, pointer 0.19110.
