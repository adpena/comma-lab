# DIRECTIVE — frozen-instance exploit of the horizon d_seg (cross-frame structure) 20260623

**Source:** operator, 2026-06-23: *"the full contest information space is known and frozen and we can
analyze and overfit to one video."* Reframes a127's "diffuse / no static always-wrong pixel" NO-GO.

## The reframe (binding)
The contest is exact-overfit to ONE frozen instance. Computable ONCE + fully known:
- the 1200 source frames (upstream/videos/0.mkv, the comma2k19 RAV4 segment),
- the frozen SegNet/PoseNet (upstream/modules.py) → the EXACT GT argmax for all 1200 frames + the EXACT
  GT pose for all 600 pairs,
- evaluate.py (the score; use tac.contest_score, NOT hand-rolled — Catalog #391).
Distortion is on KNOWN task-space targets, not pixels.

## The opening a127 left (the decisive test)
a127 measured (b42730c36, 600 pairs): residual d_seg is at the HORIZON band (rows 96-288 = 97.8%),
**content-dependent**, max per-pixel flip-frequency 5.8% → **no STATIC always-wrong pixel** → no static
$0 transform. BUT a127 tested only STATIC-per-pixel structure. Under the frozen instance, "content-
dependent" = "different per frame yet fully determined + KNOWN." The untested question:

> Is the SEQUENCE of 1200 per-frame horizon flip-sets LOW-DIMENSIONAL in the KNOWN ego-motion?

Mechanism: horizon row v_h(t) = cy + fy·tan(pitch(t)); pitch(t) is a smooth known function (comma2k19
ego-motion GT, or recover from the frontier's own stored pose). If the flip-sets track a smooth horizon
trajectory, the correction is NOT 1200 independent sets (~264KB, a127's implicit per-frame-independent
floor) but **one ego-parameterized horizon trajectory + small per-frame deviations → O(trajectory params)**,
potentially a few hundred bytes → a JUMP-INDEPENDENT pointer-move that a127 wrongly appeared to close.

## The probe (fires AFTER aa98 + a90 report; reuses their flip-measurement + byte-floor derivation)
1. Compute the KNOWN per-frame horizon flip-set for all 1200 frames (reuse aa98's exact-scorer flip
   measurement on the frontier).
2. Recover the per-frame horizon row / ego-pitch (from comma2k19 GT pose or the frontier's pose section).
3. Measure the cross-frame intrinsic dimension of the flip-set sequence vs the ego-pitch / horizon-row:
   regress flip positions against v_h(t); compute residual after the trajectory model.
4. Estimate the byte cost of the PARAMETERIZED correction (trajectory params + entropy-coded residual
   deviations) vs the break-even budget (tac.contest_score.break_even_d_seg). GO if O(few-hundred-bytes)
   for a real Δd_seg; honest NO-GO if the residual after the trajectory model is still high-entropy.

## Hard invariants
- This is the TASK-SPACE / frozen-instance / witness-compiler paradigm (CLAUDE.md "Evaluator-Equivalent
  Witness Compiler"; MEMORY "INDIRECT rate-distortion / task-space code is the prize"). Code the KNOWN
  scorer-targets, not the pixels.
- Authority: exact frozen SegNet through the eval round-trip; GT via frame_utils.yuv420_to_rgb (NEVER
  PyAV rgb24); score via tac.contest_score; any win still needs byte-close + upstream/evaluate.py.
- $0 / CPU-light; NEVER edit upstream/; no premature kill (a127's NO-GO was static-only; this is the
  cross-frame-structure re-test the frozen frame demands).
