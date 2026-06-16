# FiLM-v2 trunk decoupling — the EXACT ∂d_seg/∂(pose-objective)=0 completion

`[contest-CPU advisory]` — NO score claim. This memo documents a **gradient-routing
change** (default-OFF, byte-identical) proven correct by a gradient test on a forced-CPU
split. It does NOT move the exact frontier; it UNBLOCKS the oomph(seg)↔FiLM(pose) synergy
A/B that the parent will measure on GPU. Authority: torch-CPU TRUSTED (CLAUDE.md "local
CPU + MLX GPU good"); the in-loop d_seg/d_pose are advisory until a byte-closed
`upstream/evaluate.py` run.

Lane: completion of integration step #1 in
`.omx/research/production_readiness_bind_all_ingredients_20260616.md` (the synergy
section). Sister memory:
`feedback_score_over_training_time_always_pose_throttle_is_score_negative_20260616.md`.

## 1. The leaked-synergy diagnosis (why FiLM-v2 was INCOMPLETELY decoupled)

FiLM-v2 (`pose_film_v2.PoseFiLMHNeRVWrapperV2`) makes the FiLM **HEAD** decoupled: it
applies a residual FiLM to `rgb_0` ONLY, leaving `rgb_1` (the frame the contest SegNet
reads for d_seg) FiLM-CLEAN. So `∂d_seg/∂(FiLM-params) = 0` — perturbing the FiLM cannot
move the SegNet frame `f1` (proved here by `test_d_seg_invariant_to_film_pose_params`).

BUT the **POSE LOSS gradient** still leaked into the SHARED decoder. The contest PoseNet
is a **2-frame** network — it reads BOTH `f0` AND `f1`. So in the split-by-head backward
the pose cotangent `cot_pose` is a `(B, 2, 384, 512, 3)` gradient whose **f1 slice is
non-zero**, and `f1 = rgb_1(x)` where `x` is the SHARED trunk feature. The single fused
`decoded_bhwc.backward(gradient=cot_seg + cot_pose)` therefore flowed pose gradient into
the shared trunk + latents + `rgb_1` through TWO paths:

* `pose_loss → f1 → rgb_1 → trunk → latents`, and
* `pose_loss → f0 → rgb_0 → x0 = x + film_resid(x,cond) → trunk → latents`.

Training to reduce d_pose thus re-tuned the SHARED trunk, which moves `f1`, which moves
d_seg — the coupling FiLM-v2 was supposed to eliminate.

**Measured symptom (the leaked synergy):** under the oomph seg loss, d_pose drifts UP
monotonically (0.000335 → 0.000417 over ep10-59) as the shared trunk re-tunes toward seg.
With `∂S/∂d_pose = 5/√(10·d_pose) ≈ 85.5 ≈ 86% of ∂S/∂d_seg = 100` at the frontier
operating point, an un-arrested pose drift is nearly as score-costly as the d_seg we
optimize — so the leak is a real score cost, not a curiosity.

## 2. The fix — gradient routing (`cfg.pose_film_trunk_stopgrad`, default-OFF)

A new opt-in driver Config flag `pose_film_trunk_stopgrad: bool = False`. Default False =
the pose cotangent flows into the whole shared graph EXACTLY as today (byte-identical
gradient; the live A/B is unaffected). It is guarded in `__post_init__` to require
`pose_film_enabled` AND `pose_film_version == 2` AND `split_by_head` (a mis-config
fail-closes rather than silently no-ops).

When ON, the seg + pose cotangents are **no longer fused**. In `_split_by_head_backward`:

1. `decoded_bhwc.backward(gradient=cot_seg, retain_graph=True)` — the SEG cotangent trains
   the WHOLE graph (trunk + latents + rgb_0/rgb_1). The FiLM pose params get ~0 from seg
   (SegNet reads only the FiLM-clean f1, so the seg cotangent on f0 is exactly 0).
2. SNAPSHOT `.grad` of every SHARED (non-FiLM) param + the latents (`_non_film_grad_params`).
3. `decoded_bhwc.backward(gradient=cot_pose)` — the POSE cotangent ACCUMULATES onto ALL
   params (shared AND the FiLM pose path).
4. RESTORE the snapshot onto every shared param + latents — removing the pose contribution
   there. The FiLM pose path (`pose_mlp` + `film_resid`) is NOT restored, so it keeps the
   pose gradient.

Net: **trunk + latents + rgb_0/rgb_1 are trained by SEG only; the FiLM pose path by POSE
only** → the two objectives are orthogonal and `∂(shared)/∂(pose-objective) = 0` EXACTLY.

The mechanism is per-param-group gradient masking via separated backwards. I chose it over
a forward-detach because it is provably exact (the restored shared grad is bit-identical to
a seg-only backward, with NO residue from the pose backward) and requires NO change to the
pristine v2 wrapper (`pose_film_v2.py` is untouched). The weight-domain regularizers (C1a
entropy + Lever-1 rate surrogate) run AFTER the routing and legitimately accumulate onto
the shared trunk (they are weight-domain, not pose) — no conflict.

Files: `src/tac/torch_vehicle/driver.py` (the Config flag + `__post_init__` guard +
`_non_film_grad_params` helper + the `_split_by_head_backward` routing); the launchers wire
the flag (§5). The v2 wrapper is unchanged.

## 3. The ∂d_seg/∂pose = 0 PROOF (NO-FAKE — proven, not asserted)

`src/tac/torch_vehicle/tests/test_film_trunk_decoupling.py` (16 tests, all pass). The
decoupling is PROVEN by gradient tests, not by docstring claim. The decisive ones:

* **`test_flag_on_pose_grad_zero_on_trunk_and_latents`** (the load-bearing NO-FAKE test):
  with the flag ON, after a SEG+POSE split backward the shared params' `.grad` is
  BIT-IDENTICAL to a SEG-ONLY backward (pose residue removed) while the FiLM pose params
  carry a non-zero pose gradient that differs from seg-only.
* **`test_flag_off_pose_grad_DOES_reach_trunk`** (the behavioral contrast): with the flag
  OFF, the fused backward DOES leak pose grad into the shared trunk — proving the flag
  changes REAL behavior (not a cosmetic toggle).
* **`test_flag_on_vs_off_shared_grad_differs_film_path_active`**: same init, ON vs OFF →
  the latents grad differs (pose removed ON), confirming the routing has effect.
* **`test_d_seg_invariant_to_film_pose_params`** + negative control
  **`test_film_perturbation_DOES_change_f0`**: perturbing the FiLM pose params leaves f1
  (hence d_seg) bit-identical while changing f0 (the FiLM is alive, not a vacuous pass).
* **`test_full_film_pose_path_receives_grad_off_identity`**: off the wrapper's identity
  init the ENTIRE pose path (pose_mlp.fc1/fc2 + gamma_head + beta_head + proj) receives a
  non-zero pose gradient (the routing reaches the whole subgraph).
* **`test_seg_still_trains_trunk_under_flag`**: seg STILL trains the shared trunk under the
  flag (we did not freeze the trunk entirely).
* Config guards: default-OFF, requires-film, requires-v2, requires-split, valid-ON.

**Empirically measured per-param grad pattern under the flag (forced-CPU split, ch=8):**
shared trunk (stem/blocks/skips/refine) + `rgb_1` + latents all carry nonzero grad (SEG);
`rgb_0` carries ZERO grad (see §4); `film_resid.proj` + `film_resid.beta_head` carry the
pose gradient. (`pose_mlp` + `gamma_head` carry zero ONLY at the identity init transient —
∂cond/∂beta = beta_head.weight = 0 at zero-init — and engage after one step off identity,
verified by the off-identity test.)

## 4. Design consequence to flag: `rgb_0` freezes w.r.t. pose under the flag

A subtle, intended consequence: `rgb_0` (the pose-conditioned frame's head) gets ZERO grad
from seg (SegNet does not read f0) AND its pose contribution is masked out under the flag —
so `rgb_0` is trained by NEITHER objective and FREEZES at its warm-start value. This is
consistent with the design ("the FiLM pose path is the only place pose grad lands; the
shared decoder, rgb_0 included, is the seg-trained capacity") — the pose-conditioned motion
is carried by `film_resid` modulating rgb_0's INPUT, not by adapting rgb_0's weights. It
is, however, an extra constraint vs the current v2 (where rgb_0 IS pose-trained), and feeds
directly into the measured question below.

## 5. The MEASURED-QUESTION caveat (this is an A/B input, NOT a sure win)

Freezing the trunk (and rgb_0) w.r.t. pose means the FiLM head + the ~6 stored pose
scalars/pair must carry ALL the pose signal. This MIGHT hold d_pose WORSE (if the scalars +
the bounded residual FiLM are insufficient to express the per-pair motion) OR BETTER (no
seg/pose tug-of-war in the shared trunk; the measured ep10-59 d_pose drift is the cost of
the tug-of-war that complete decoupling removes). The Quantizr design (store-6-pose-scalars
+ FiLM dual head) is evidence the scalars + FiLM suffice, but **it is empirical for THIS
substrate**. Therefore this is an OPT-IN mode for a GPU A/B, NOT a forced default. The
parent must A/B it and confirm d_pose is HELD (not regressed) before adopting.

## 6. Recommended A/B (the parent's GPU run)

Two arms, identical warm-start + seed + curriculum + `--pose-grad-adaptive` (pose every
epoch per the score-over-time directive — the throttle is a safety net, not a speed lever),
differing ONLY in the flag:

* **Arm CONTROL (current v2):** `--pose-film-v2 --split-by-head --train-device mps`
  (+ the oomph seg crank). The coupled FiLM-v2 (pose still leaks to the trunk).
* **Arm DECOUPLED (this completion):** add `--pose-film-trunk-stopgrad`.

Acceptance for DECOUPLED to be adopted as the production default: under the oomph seg
crank, DECOUPLED must (a) HOLD d_pose (no monotonic ep-over-ep drift up; ideally flat at
the stored-pose floor), AND (b) NOT regress d_seg vs CONTROL (it should be ≤, since the
trunk is now seg-only). If DECOUPLED holds d_pose AND d_seg descends at least as fast, the
leaked-synergy is fixed and the oomph(seg)↔FiLM(pose) orthogonality is unlocked. If
DECOUPLED regresses d_pose materially, the 6 scalars are insufficient → keep CONTROL and
investigate a richer pose carrier (more stored scalars / a small pose-trainable rgb_0
sub-path) as a follow-up. Both arms are byte-closed + dual CPU/CUDA evaluated before any
score claim.

Launcher (the decoupled-oomph run): `experiments/launch_oomph_finetune_disambiguator.py`
gains `--pose-film-trunk-stopgrad` (mirrors `--pose-film-v2` threading; printed into the
run manifest JSON for observability). `experiments/launch_taper_ab.py` surfaces the flag
for argparse-consistency but it is INERT there (taper is non-FiLM by design; passing it ON
fail-closes at the Config guard).

## 7. Solver wire-in hooks (Catalog #125)

1. **Sensitivity-map** — N/A (research_only advisory; this is a gradient-routing change, no
   per-byte sensitivity row).
2. **Pareto constraint** — N/A directly; the completion REMOVES a parasitic pose↔seg
   coupling so the seg and pose axes are independently optimizable (a precondition for a
   clean joint-Lagrangian Pareto step, not a constraint itself).
3. **Bit-allocator hook** — N/A (no per-tensor importance change; FiLM ships in the decoder
   blob as before).
4. **Cathedral autopilot dispatch** — N/A (default-OFF flag; not archive-deployable until
   the A/B promotes it).
5. **Continual-learning posterior** — the A/B's measured d_pose-hold verdict is the anchor
   to land (the parent emits it on the GPU run; this memo is the design anchor).
6. **Probe-disambiguator** — the A/B (CONTROL vs DECOUPLED) IS the disambiguator for the
   open question "do the 6 stored scalars carry enough pose signal?"; this memo specifies
   it (§6).

## 8. Byte-identical-default confirmation

Flag OFF: `_split_by_head_backward` takes the EXACT legacy fused
`decoded_bhwc.backward(gradient=cot_seg + cot_pose)` path (the throttled path is `cot_seg`
alone, unchanged). The 49-test regression (`test_driver_resume` + `test_split_by_head_grad`
+ `test_pose_grad_throttle` + `test_adaptive_pose_controller`) + 33-test
faithful/v2-wrapper suite all pass unchanged. `test_forward_parity_flag_on_vs_off` confirms
the FORWARD (rendered frames / archive bytes) is bit-identical ON vs OFF (the flag is a
backward-routing change only). The default Config has `pose_film_trunk_stopgrad=False`.
