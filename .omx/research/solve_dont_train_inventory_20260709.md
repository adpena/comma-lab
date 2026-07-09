# SOLVE-DON'T-TRAIN inventory (#342) — 2026-07-09

**Task:** enumerate every witness-system block that is SOLVABLE (linear / quadratic /
KKT / closed-form / OT / eikonal) rather than trained, with a per-block solve schedule
(where / when / conditions / readiness). $0, read-only, no GPU.

**STORES CONSULTED:** CLAUDE.md §OPERATOR-PRIORITY + §Meta-Lagrangian/Pareto solver;
`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (grep
solve|KKT|waterfill|closed-form); `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`;
MEMORY.md L77 (#341 quadratic head chart), L68 (pose banked), L71 (analytic lane band),
L57/L66 (annulus/margin-saliency); canonical equation
`quadratic_head_chart_subset_solve_gap_v1`; source files cited inline (file:line).

**LABELS:** MEASURED = through-R n600 or offline probe with artifact · DERIVED = math
property of the form · ASSUMED = plausible, unverified. **verdict_scope** on every negative
per the FORMULATION-not-FAMILY ladder.

---

## THE INVENTORY

| # | Block | Math form | Replaces (train-steps saved) | WHERE | WHEN valid (conditions) | Prereq artifacts | Solve wall-clock | Readiness / verdict_scope |
|---|---|---|---|---|---|---|---|---|
| 1 | **Terminal head finisher (#341)** — the affine head `out_sdf.{weight,bias}` + `out_tex.{weight,bias}` + `palette` (~791 params) | **Gauss-Newton / damped Newton-CG** (Levenberg λ 0.1→0.033→0.011, 16 CG iters, HVP=vjp-of-grad) on exact tau-stage loss | terminal fine-tune stage (the last curriculum leg's epochs on the head) | **terminal** (tau-best ckpt) | basin reached — head chart **near-quadratic CONFIRMED** (LM ρ 0.847/0.868, MEASURED); **full-P (P=600) ONLY** — K=8 subset OVERFITS (+5.1% n600) | `levelset_witness_ema_BEST.npz` tau-best; exact tau-stage loss (τ=0.3 softplus + 0.001·length); HVP | ~11 min/CG-iter GPU@17× × ~16 ≈ **3 h GPU** (19 s/pair·HVP CPU → 3.2 h/iter CPU) | **HIGH** (GO issued). verdict_scope: **subset-solve = FORMULATION NO-GO** (measured); full-P solve UNTESTED = the GO. NOT $0. |
| 2 | **Per-class head bias offsets b_c (#288/#218)** — Laguerre / power-diagram reweight of argmax cells | **damped-Newton semi-discrete OT** (Kitagawa-Merigot-Thibert 2019; K×K softmax-cov pinv, zero-sum gauge, Armijo backtrack, terminal-quadratic) | trained per-class bias correcting minority-class (Lane 0.59% / Movable 1.56%) erasure — ~57% of flips are Lane↔Road | **decode / init** — folds into `out_sdf.bias` (5 floats) → **BYTE-FREE** | any time phi field (N,K) available; solves b* s.t. soft cell masses == target GT class freqs | phi field `(N,K)`; `target_masses` (GT class frequencies) | **<1 s CPU** (64 Newton iters, K=5) | **HIGHEST $0-READY.** Full solver BUILT (`laguerre_logit_offset.py:177`) but **UNWIRED** — trainer wires only the Menon `-τ·log(π)` heuristic INIT (`:4378`). Orphaned high-value → duty-to-measure. |
| 3 | **KKT reverse-waterfill bit allocation (#157)** | **CLOSED-FORM KKT** (separable convex; `b_t* = log₂(λ·ln2·c_t/numel_t)`, `c_t=s·absmax·√numel`) | hand-tuned per-tensor bit sweep | **export / decode** (archive RATE term) | after per-tensor sensitivities measured; λ traces the RD curve | `CombinedTensorSensitivity` (per-tensor sens+absmax+numel) | **<1 s** | **DONE — cite, don't rebuild.** `frontier_exact_bitalloc.py:331`. Per-dim sister `mod_dim_dynamics.py:229`. |
| 4 | **Horizon poly-fit** (road/sky boundary) | **closed-form `np.polyfit` (lstsq, deg 1–2)** + robust trim-refit | learning the horizon boundary (geometric generator; the "today 14.6× SOLVE not train" class) | **decode / prior generator** (rule-118 FREE) | always (per-frame from lstar top-contiguous sky run); ≥ deg+2 sky cols | `lstar` argmax (or GT sky) | **ms/frame** | **READY.** `road_horizon_component.py:245` `fit_horizon_line`. Rate/prior side, not a d_seg train-lever. |
| 5 | **SE(3) B-spline pose-ξ fit** | **least-squares control-pose fit** (sample+refit over cumulative twist; deterministic) | learned pose-trajectory params → compress ξ to M control poses | **compress-time** (pose carrier RATE) | after dense ξ estimated; picks smallest M below tol via fit-error curve | `dense_xi (n_pairs,6)` + `tac.lie` oracle | **ms** | **READY.** `ego_xi_trajectory.py:421` `fit_se3_bspline_controls` + `bspline_fit_error_curve`. Pose already BANKED (R1 dxi, MEMORY L68) → this is the rate-compression of an already-solved axis. |
| 6 | **Latent Gauss-Newton seed** (per-video compress-time) | **Gauss-Newton in 28-dim latent**, decoder frozen (per-pixel Jacobian projects latent toward argmax cell) | per-video overfit refine epochs (amortized-init #211 lens) | **init / compress-time per video** | frozen decoder; per-video | frozen decoder + target argmax | probe-scale | probe-level (`probe_compress_time_seed_and_solve_dseg.py:153` `latent_solve`). verdict_scope: probe, not n600-validated. |
| 7 | **Init-time SDF redistance** | **distance-transform** (labels → signed distance) | initializing phi as a valid SDF vs learning it from noise | **init** (once) | at model init from seed labels | seed labels | ms | **DONE (init only).** `lever_b_levelset_generator.py:303` `signed_distance_fields`; `lane_sdf_component.py:302`. |
| 8 | **Curriculum stage-transition detector** | **lstsq slope** of verdict trajectory | hand-set stage-boundary thresholds | **stage-boundary** (control plane) | ≥2 window points | verdict history | µs | S-NEUTRAL control solve (`train_levelset…py:2413` `_cl_lstsq_slope`). Not a weight solve. |

---

## HONEST NEGATIVES (verdict_scope: FORMULATION / DERIVED)

- **FiLM per-pair params given frozen trunk ≠ least-squares.** DERIVED: the #341 anchor
  EXCLUDES FiLM gains explicitly ("FiLM gains excluded — not affine"). FiLM's multiplicative
  modulation makes the trunk→phi map **non-affine in the FiLM params**, so a
  frozen-trunk-then-solve-FiLM step is NOT a clean lstsq/quadratic solve. **verdict_scope:
  FORMULATION-level** — only the *additive head* (`out_sdf/out_tex/palette`) is affine-solvable;
  the FiLM path stays trained. (A local GN step around the FiLM operating point is possible but
  is a linearization, not a closed-form solve — do not claim it as one: NO-FAKE.)
- **Live eikonal re-init by FMM is NOT built.** The eikonal terms in the trainer
  (`train_levelset…py:1406–1556, 5255` `_eikonal_length_mlx` / `_eikonal_steik_mlx` / …) are
  **LOSS terms (gradient descent to |∇m|→1)**, not a fast-marching solve. A live FMM/redistancing
  re-init (replacing eikonal-loss epochs with an O(N log N) solve at stage boundaries) is a
  **CANDIDATE, not a ready block** — only the *init-time* distance-transform SDF (row 7) exists.
  verdict_scope: ASPIRATIONAL — flag for build, do not cite as solved.

---

## RANKING (train-time-saved × d_seg/d_pose-impact × readiness)

1. **#341 full-P terminal head GN/CG (row 1)** — TOP by impact × readiness (the GO). Replaces
   the whole terminal head-finetune leg; direct d_seg; basin CONFIRMED near-quadratic. NOT $0
   (~3 h GPU).
2. **#288 damped-Newton OT head offsets b_c (row 2)** — TOP $0-IMMEDIATE. Full solver built but
   UNWIRED; byte-free; attacks the dominant Lane↔Road erasure; <1 s. Highest readiness-per-dollar.
3. **#157 waterfill (row 3)** — already SOLVED (rate axis); reuse.
4. **Horizon poly / SE(3) B-spline (rows 4–5)** — ready geometric/rate generators; pose side banked.

## TOP SOLVE TO BUILD NEXT + EXACT FIRING CONDITION

**Build: the #341 in-trainer full-P Gauss-Newton / damped-Newton-CG head finisher** (matches
the standing GO). **Firing condition (all must hold):**
1. **Stage = terminal** — tau-best EMA checkpoint reached (end of the τ-softplus leg, the
   `levelset_witness_ema_BEST.npz` the probe anchored on).
2. **Basin verified** — a cheap re-verify LM gain ratio ρ ∈ ~[0.8, 1.2] on the CURRENT checkpoint
   (the head chart is near-quadratic HERE, not assumed from ep650).
3. **Full-P (P=600) ONLY** — solve over ALL 600 pairs; **NEVER a K<P subset** (K=8 measured
   +5.1% n600 overfit; the transfer law `net=(K/P)·Δin+(1−K/P)·Δout` pins it).
4. **Params = the ~791 affine head** (`out_sdf.{w,b}` + `out_tex.{w,b}` + `palette`); FiLM gains
   EXCLUDED (non-affine).
5. **Solver = damped Newton-CG** (Levenberg λ 0.1→0.033→0.011, ~16 CG iters, HVP = vjp-of-grad)
   on the **exact tau-stage loss** (verified vs launch argv), MLX-GPU with `--fused-r-kernel` for
   bit-identity (MEMORY L70) or CPU authority.
6. **Verdict through R** — the resulting head folded in, byte-closed, d_seg re-measured on the full
   n600 through the exact R + frozen CPU SegNet (advisory→authority per axis).

**$0 companion to fire FIRST (row 2):** wire `damped_newton_ot_offsets` (replacing the Menon
`-τ·log π` heuristic at `train_levelset…py:4378`), solve b* against GT class masses on the current
phi field, fold byte-free into `out_sdf.bias`, and rank the argmax-disagreement proxy — then confirm
through R. This is the immediate, no-GPU, high-value move while the #341 GPU solve is scheduled.

---

*NO-FAKE: every row cites a real code path; rows 1–3 are MEASURED/DONE, rows 4–8 are BUILT
generators, the two negatives are DERIVED/ASSUMED and labelled. No aspirational block is claimed
solved.*
