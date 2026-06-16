# Adaptive Pose-Gradient Controller (APGC) — closed-loop pose-throttle replacement

**Date:** 2026-06-16T19:42:32Z
**Authority:** `[contest-CPU advisory]` — NON-PROMOTABLE. NO score claim. This is a
train-time wall-clock + drift-arrest lever; the exact d_seg/d_pose that pick BEST still
run the full scorer on the CPU authority. Default-OFF is byte-identical to today.
**Scope:** `src/tac/torch_vehicle/driver.py` (config + state + controller + checkpoint
persistence), `src/tac/torch_vehicle/checkpoint.py` (3 persisted keys),
`experiments/launch_oomph_finetune_disambiguator.py` + `experiments/launch_taper_ab.py`
(4 new flags, default-off), `src/tac/torch_vehicle/tests/test_adaptive_pose_controller.py`
(19 NO-FAKE tests).

## The bug the controller fixes

The split-by-head PoseNet path is ~51% of each epoch (CPU FastViT fwd+bwd, measured
10.2s of a 20s epoch @96 pairs). To reclaim it a **static** throttle was added
(`pose_grad_every_k=4` + `pose_grad_resume_threshold=0.001`): compute pose only every
k-th epoch, force-compute while pose_mse exceeds the threshold.

The static throttle is **open-loop** and fails on a drifting pose:

- Under the oomph seg crank, d_pose DRIFTS UP through the **shared decoder trunk** as it
  re-tunes toward seg. Measured monotonic over ep10-40: `0.000335 → 0.000363 → 0.000389
  → 0.000398 → 0.000408` (`[contest-CPU advisory]`).
- The fixed resume threshold (0.001) is ~2.5-3× the actual pose_mse (~0.0004), so it
  **NEVER fires** — pose is corrected only 1-in-4 epochs WHILE drifting.

### The equimarginal / constraint grounding (why the drift is score-costly)

The contest score is `S = 100·d_seg + √(10·d_pose) + 25·bytes/N`. The pose term is
nonlinear, so its marginal sensitivity blows up as d_pose → 0:

```
∂S/∂d_pose = 5 / √(10·d_pose)
```

At the frontier operating point d_pose ≈ 0.0004:

```
∂S/∂d_pose = 5 / √(10 · 0.0004) = 5 / √0.004 = 5 / 0.0632 = 85.5
∂S/∂d_seg  = 100   (constant)
```

So **un-arrested pose drift is ≈ 86% as score-costly per unit as the d_seg we are
optimizing.** The measured ep10-40 drift (Δd_pose ≈ +7.3e-5) already cost
≈ `85.5 · 7.3e-5 ≈ +0.006 S` — a real, un-recovered frontier debt incurred WHILE the
throttle thought pose was "solved."

The fix is the **equimarginal principle**: spend the expensive pose path only when the
marginal score harm of NOT correcting (pose above the deadband, or rising) exceeds the
wall-clock saved by skipping — i.e. hold d_pose at its (moving) floor with minimum spend.

## The controller spec

Config fields (ADDITIVE; defaults reproduce today byte-identically):

| flag | default | meaning |
|---|---|---|
| `pose_grad_adaptive` | `False` | master switch. False → the EXISTING static k/threshold branch runs UNCHANGED (byte-identical). True → static k/threshold are IGNORED; APGC governs `do_pose`. |
| `pose_grad_floor_tol` | `0.08` | deadband: hold pose ≤ `floor·(1+tol)`. 8% ≈ a `0.08·85.5·0.0004 ≈ 0.0027` S slack band (≈ 0.0023 at the deeper floor). |
| `pose_grad_k_max` | `8` | sparsest cadence at floor AND the measurement-floor (force a pose MEASUREMENT after this many blind epochs). |
| `pose_grad_trend_window` | `3` | # recent COMPUTED pose_mse for the derivative/slope term. |

`__post_init__` (adaptive only): `tol > 0`, `k_max ≥ 1`, `trend_window ≥ 2`, AND
`split_by_head=True` (the throttle is split-only) — else raise.

Tracked driver state (persisted in `_capture_state` / restored in `_restore_into` /
written by `save_checkpoint`; all default on the non-adaptive path → byte-identical):

- `_pose_floor` — running MIN of every computed training pose_mse (the adaptive floor).
- `_pose_mse_hist` — last `trend_window` computed pose_mse (the slope term), trimmed.
- `_last_pose_epoch` — global epoch of the last pose COMPUTE (measurement-floor ref).

Decision (pure helper `_adaptive_do_pose`, unit-testable in isolation):

```
if pose_floor is None or last_pose_mse is None:
    do_pose = True                                   # establish floor / first epoch
else:
    dev = last_pose_mse / pose_floor                 # rel deviation from the floor
    rising = len(hist) >= 2 and hist[-1] > hist[0]   # the derivative term
    if dev > 1 + tol or rising:
        do_pose = True                               # drift-arrest: every epoch
    else:
        slack = clamp((1 + tol - dev) / tol, 0, 1)   # 1 at floor → 0 at band edge
        k = max(1, min(k_max, round(1 + slack*(k_max-1))))   # proportional cadence
        do_pose = (epoch % k == 0)
if epoch - last_pose_epoch >= k_max:                 # measurement-floor
    do_pose = True
```

After the backward, when pose was actually COMPUTED, the controller advances: update the
running-min floor, append to the trend window (trim), stamp the measurement epoch. Skipped
epochs leave the state stale BY DESIGN.

## Known limitation (measurement vs correction)

A skipped epoch does **not MEASURE** pose — the pose forward is fused with the backward
here, so when we skip the backward we also skip the measurement. The **measurement-floor**
(force a compute every `k_max` epochs) bounds how long the controller may run blind. A
future refinement is a cheap forward-only pose probe decoupled from the backward (measure
without paying the full bwd), which would let the controller observe drift on every epoch
while still throttling the expensive cotangent.

## Tests (19, all passing)

`src/tac/torch_vehicle/tests/test_adaptive_pose_controller.py`:
- config validation (adaptive requires split_by_head; tol/k_max/window bounds; defaults off);
- floor establishes on first epoch; running-min floor invariant;
- drift-arrest fires when dev > 1+tol;
- proportional cadence monotone in deviation depth (deeper → sparser);
- trend override (rising within band → compute; flat → defer to cadence);
- measurement-floor forces a compute after k_max blind epochs;
- **default-off uses the STATIC cadence + leaves controller state untouched** (byte-identical);
- **decisive no-silent-revert**: adaptive ON diverges from the static k cadence AND the
  controller state actually advances;
- **checkpoint round-trip**: a real kill+restore persists `_pose_floor`/`_pose_mse_hist`/
  `_last_pose_epoch` through `save_checkpoint` → `load_checkpoint` → `_restore_into` (NOT a
  bypass), plus a legacy-checkpoint-without-keys → defaults backward-compat test.

Regression: `test_driver_resume.py` + `test_pose_grad_throttle.py` (22 tests) pass
unchanged; `test_split_by_head_grad.py` + `test_export_and_faithful.py` (14) pass. The
`checkpoint.py` change adds 3 keys to the saved blob (sister of `ema_step` /
`tensor_sensitivity_ema`); backward-compatible (absent keys → defaults).

## Recommended A/B + production flags

```
# Production / recommended:
--pose-grad-adaptive --pose-grad-floor-tol 0.08 --pose-grad-k-max 8

# A/B (apples-to-apples; same warm-start + same seed, differ only by the throttle):
#   arm STATIC  : --pose-grad-every-k 4 --pose-grad-resume-threshold 0.001   (today)
#   arm ADAPTIVE: --pose-grad-adaptive --pose-grad-floor-tol 0.08 --pose-grad-k-max 8
```

Both launchers (`launch_oomph_finetune_disambiguator.py`, `launch_taper_ab.py`) thread the
4 flags (default-off). The expected win: ADAPTIVE arrests the ep10-40 d_pose drift (holds
it at floor) for ≈ the same pose-compute budget the static throttle spends mis-allocated
across non-drift epochs — recovering the measured ≈ +0.006 S drift debt at no extra
wall-clock, with the measurement-floor capping drift-blindness.
