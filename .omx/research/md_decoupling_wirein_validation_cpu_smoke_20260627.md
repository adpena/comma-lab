# MD-Decoupling wire-in + validation + $0 CPU smoke (2026-06-27)

Operator directive 2026-06-26 "Wire in MD-Decoupling". $0 CPU-only (Metal GPU
reserved for the live witness run — every arm ran `--mlx-device cpu
--verdict-device mlx-cpu`, authority = frozen CPU-torch SegNet/PoseNet, NO-FAKE
realized d_seg/d_pose through the R operator). Optimizer:
`src/tac/optimization/md_decoupling.py` (arXiv:2606.25971). Trainer:
`experiments/train_witness_realized_through_R_mlx.py --optimizer {adamw,md}`.

## CORRECTNESS (paper vs code) — VALIDATED, no deviation, no fix needed
Math checked line-by-line against the paper/memo:
- Reparam `W = diag(gr) @ What @ diag(gc)` = elementwise `gr[:,None]*What*gc[None,:]`. ✓
- Chain-rule grads (G = dL/dW): `g_what = G*gr⊗gc`, `g_gr = Σ_j G·What·gc`,
  `g_gc = Σ_i G·What·gr`. Independently confirmed against `mx.grad` AND via a new
  end-to-end descent test (below). ✓
- Separate LRs: direction `eta_W = lr`, gains `eta_g = gain_lr_scale·eta_W`
  (default 1/3, paper 3e-3/1e-3). ✓
- Hypersphere projection: `What *= ||W_init||_F / (||What||+1e-12)` after each
  direction step; gains init 1.0 → identity at step 0. ✓
- Weight-decay dropped; warmup schedule honored but no-op for direction. ✓
- **Muon fidelity:** `newton_schulz5` (coeffs 3.4445/-4.7750/2.0315, pre-norm,
  transpose-wide) AND `_muon_update` momentum convention
  (`v = m·v + (1-m)·g`; nesterov `g·(1-m)+v·m`; `max(1,rows/cols)**0.5` scale)
  are **byte-faithful to the INSTALLED `mlx.optimizers.Muon`** (verified by
  `inspect.getsource`). The docstring claim is honored.
- Gains always Adam; direction Adam or Muon. ✓ `code` (video-derived latent) and
  biases route to plain Adam (not decoupled) — correct.

Tests: `src/tac/optimization/tests/test_md_decoupling.py` — **13/13 pass**
(12 pre-existing behavioral + 1 NEW `test_md_step_chain_rule_descends_to_target_weight`
that drives the real `_md_step` to minimize `0.5||W−W*||²` to <1e-3·L0 — fails if
any of the 3 gradient formulas is wrong; also re-asserts the sphere invariant).

## WIRE-IN — complete, no gaps
`--optimizer md` constructs `MDDecoupledOptimizer(learning_rate=build_lr_schedule(),
gain_lr_scale=args.md_gain_lr_scale, base=args.md_base)`. `default_md_eligible`
(2D `*.weight`) routes ALL witness MLP weight matrices through MD:
`in_proj.weight`, `film.weight`, `hidden.{0..N}.weight`, `out.weight`. Biases +
`code` latent → plain Adam (correct: `code` is video-derived payload, not a
weight). `_B` Fourier table is `_`-prefixed → excluded from `model.parameters()`
(MLX convention) → never seen. EMA / byte-close / verdict untouched (model tree
stays materialized W). Confirmed engaged: smoke logged
`{"stage":"optimizer","kind":"md_decoupling","md_base":"adam","md_gain_lr_scale":0.333}`.

## $0 SMOKE — gt_n6 cache, render 64×96, seed 0 (deterministic; parallel CPU OK)
Matched run = margin curriculum engage at ep12 (the stage transition), lr 1e-3,
n_restarts 2, ONLY optimizer differs:

| arm | optimizer | gnorm_max | divergence_restarts | best d_seg_live(EMA) | best d_pose_live |
|---|---|---|---|---|---|
| arm_adamw | adamw 1e-3 | **10869.6** | **2** | 0.235 | 2.59 |
| arm_md_par | md adam 1e-3 | **376.5** | 1 | 0.5075 (pinned) | 49.62 |
| arm_adamw_ctrl (no-margin) | adamw 1e-3 | 6919 | 0 | 0.497 | 9.48 |
| arm_md_hilr (no-margin) | md adam 5e-3 | 4619 | 0 | 0.5075 (pinned) | 93.44 |

Findings:
1. **STABILITY (confirmed):** at matched lr 1e-3 through the stage transition MD
   kept gnorm bounded ≤376 while AdamW exploded to **10869** (≈29×) and tripped
   `restart_divergence` TWICE (non-EMA live weights collapsed back to baseline
   d_seg 0.507 — the EMA-shadow-lag pattern; EMA shadow had reached 0.235). MD's
   anti-collapse property is real and MEASURED. spike-guard `n_skips`=0 for all
   arms (grad-clip 1.0 + recalibration absorbed it; the DIVERGENCE detector, not
   the spike-guard, is what fired).
2. **REALIZED PROGRESS (inconclusive / MD under-steps):** at adamw's lr MD's
   d_seg stayed pinned at baseline (no pixel flips in 12 ep) and pose descended
   less; AdamW descended faster (d_seg_live 0.235, pose 2.59) but unstably.
   Raising MD lr to 5e-3 did NOT unlock d_seg descent — it just raised gnorm to
   4619. So "MD trains d_seg lower/faster" is NOT supported at this tiny $0
   scale; the bounded-update reparam under-steps within the budget.
3. **STAGE TRANSITION (the key test):** at n6 the margin-engage itself did not
   reproduce the live-run 648–772 gnorm spike (scale-dependent; AdamW's
   instability appeared in the uniform phase too). MD removes the gradient
   explosion but does NOT by itself convert the transition into clean
   convergence — it still trips the divergence detector (on no-d_seg-progress),
   and the divergence-detector / EMA-lag / under-stepping interplay is tuned for
   AdamW.

## RECOMMENDATION — PARALLEL ABLATION ARM, not a blind drop-in
Use MD as a **parallel arm**, NOT the optimizer of the decisive resumed
margin-finetune. The manual temp-anneal + LR-re-warmup AdamW fix remains the
primary path. Rationale + RISK (honest): MD's stability win is real and directly
targets the stage-transition gradient explosion, BUT at the inherited lr it
under-steps d_seg (would likely produce slow/no realized-d_seg descent within
the single-GPU budget), a higher lr (5e-3) did not help, and the trainer's
divergence detector + spike-guard recalibration + EMA cadence are AdamW-tuned.
Blind MD on the decisive run = HIGH risk of a wasted GPU run. The MD arm needs,
at real scale (n96/n600): its own lr sweep, relaxed/disabled divergence detector,
and longer budget; promote ONLY on a byte-closed exact-eval row beating the
AdamW manual-fix arm.

Evidence: `experiments/results/md_decoupling_cpu_smoke_20260627/arm_*.log`
(gitignored scratch; npz checkpoints deleted as rebuildable; logs retained).
Cross: DAG FEED-bu/bv; `feedback_different_stages_need_different_treatment_*_20260626.md`;
`reaudit_refounding_and_md_decoupling_20260626.md`.
