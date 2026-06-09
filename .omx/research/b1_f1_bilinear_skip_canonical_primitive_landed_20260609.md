# F1 LANDED — PR95 HF-residual as a GATED CANONICAL PRIMITIVE (cross-vehicle Mistake-A fix)

UTC 2026-06-09 · claude · commits `11b15cd02` (impl+tests) + `e07ff7944` (ablation launcher).
`[macOS-MLX research-signal]` / `mechanism_update_eligible` only — NO score claim, NO promotion.
Closes the diagnostic arc (F5 plateau confirmed) and lands the fix the deep-review subagent ranked #1,
built per the operator's "fractally optimized + no duplicative code + new primitives + gating" directive.

## The decision chain (each step authority-classified)
1. **F5 (telemetry_proxy, mechanism-only):** the CLEAN faithful 8-stage PR95 curriculum + Muon + ~2900
   epochs holds `seg_axis_train_loss_proxy` FLAT at mean 1.595 (std 0.015) within the single dominant
   `l7_softplus` stage; slope **2.4e-7/epoch → +0.0005 total drift over 1878 epochs** (incl. 433
   Muon-final). Verdict **ARCHITECTURAL_PLATEAU_NOT_UNDERTRAINING**. (Retracted an in-progress
   "single-stage" suspicion: `loss_form` shows all 4 stages ran — `stage_name` was a coarse wrapper.)
   Artifact: `b1_clean_pr95_f5_undertraining_control.v1.json`.
2. **Cross-vehicle audit (sister subagent, committed `9538bb4f3`):** the whole NeRV fleet is the SAME
   skip-free `PixelShuffle(sin(w=30·conv))` decoder under ~10 names, failing for TWO orthogonal shared
   mistakes — **A (architecture): no residual HF path** (no bilinear-skip, no PE, no terminal refine);
   **B (objective): the shared MLX harness base term is recon-MSE with SegNet/PoseNet distill weights
   defaulting to 0.0** → MSE rewards the mean-field. Decisive receipt: SNeRV's FAITHFUL MFU/HFR/TUB
   renderer at ep22399 = d_seg 0.711 with `observed_segnet_distillation_weight=None` (recon-only).
   SNeRV's architecture is REAL (exact DWT adjoint); it was starved by Mistake B + a mean-collapsed
   finest skip. Memo: `snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609.md`.

## What F1 fixes, and the clean separation of concerns
F1 addresses **Mistake A** (architecture). The **recon-fit probe is the clean Mistake-A isolation
test**: it is ALWAYS pure recon-MSE, so a PSNR break there is purely the residual-path effect, with
Mistake B (the objective) held constant. (Mistake B is a separate config-only fix — see "next".)

## The implementation (gated canonical primitive — no duplicative code)
Per the operator directive, the HF-residual COMPOSITION is a **shared canonical kernel**, not copy-paste
per carrier; each carrier keeps its OWN quant-aware conv forward (fractal-per-method).
- **NEW canonical kernels** in `tac.framework_agnostic.canonical_kernels` (numpy reference + MLX/torch/
  tinygrad + cross-backend parity, Catalog #383-routed): `bilinear_skip_residual_canonical(shuffled,
  identity, sin_frequency)` = `sin(w·(shuffled+identity))`; `terminal_hf_refine_canonical(h, refine_act,
  scale)` = `h + scale·sin(refine_act)`. Both fail closed on channel/shape mismatch (the channel-match
  bug class) and are MLX-gradient-reachable (verified: ∂=w·cos at 0 for both branches).
- **HiNeRV carrier** (`hi_nerv/architecture.py` + `mlx_renderer.py`): new gated config
  `use_bilinear_skip: bool=False` (+ `refine_residual_scale=0.1`). `_UpBlockMLX` gains a 1×1
  channel-match skip; the renderer gains a terminal refine conv. Both route the composition through the
  canonical kernel via thin `_bilinear_skip_residual` / `_terminal_hf_refine` wrappers.
- **Gating = zero regression:** when OFF, NO skip/refine modules are created (params 304,922,
  byte-identical legacy path); when ON, +9,532 params, same output shape, and init output std jumps
  **0.0001 → 0.0084 (84×)** — the skip injects the spatial variance the skip-free carrier lacks
  (the one-class-flat → d_seg≈0.50 escape, at init). Export is **fail-closed (NotImplementedError)** when
  ON (research-only recon-fit surface; export layout + PyTorch-oracle parity is a gated follow-up — NO
  silent incomplete archive).
- **Tests:** 6 canonical-kernel tests (numpy math, w=1 PR95-implicit, zero-scale identity, fail-closed
  ×2, MLX parity ×2, gradient-reachability) + 3 carrier-gate tests (OFF byte-identical, ON
  forwards+params+variance, ON export fail-closed). Catalog #383 gate green (21).

## The launched falsification (running now)
`scripts/launch_b1_f1_skip_recon_fit_ablation.sh` (detached) runs two skip-ON arms SEQUENTIALLY at
N=600 ep800 vs the skip-OFF control (PID 24818, plateau **21.74 dB**):
- **Arm A — skip ON, w=30** (single-variable skip add at our current frequency).
- **Arm B — skip ON, w=1** (PR95-faithful; tests H4: w=30 on a coherent carrier inside sin aliases it).
**Falsifiable prediction (subagent F1):** PSNR breaks 21.74 → >28 dB within the same budget. If Arm A
stays ~21.7 but Arm B breaks, H4 was binding (w=30 was the trap, not just the missing skip). If BOTH
stay ~21.7, H1 is falsified at the recon surface and the binding constraint is grid-PE (F2) or the
objective (Mistake B). Arms write `recon_fit_f1_skipON_{w30,w1}_<utc>/recon_fit_f1_*.json`.

## Route (authority-disciplined)
- This whole chain is advisory/mechanism — it directs the next experiment; it does NOT touch the score
  roadmap (only a contest-axis `exact_evaluate` row does). The recon-fit probe is contract-free,
  needs no export/oracle parity, no archive — the cheapest path to confirm/refute H1.
- **Next after the arms land:** (a) if PSNR breaks → wire export + PyTorch-oracle parity for the skip
  (lift the fail-closed guard), then retrain the score-aware curriculum WITH the skip and score the
  LIVE render's d_seg (the evaluator metric, not PSNR); (b) **Mistake B (orthogonal, config-only,
  cheapest fleet-wide win the audit found):** retrain SNeRV path-B with `segnet/pose_direct_live_
  distillation_weight` ON + recon annealed to a small anchor + `official_skip_high_mode='full'`
  (prediction d_seg 0.71 → <0.2; NO architecture change).
- **DO NOT** branch to codebooks / PR110++ / optimizer exotica until a carrier reaches evaluator
  fidelity (none has yet).

## Cross-refs
`b1_carrier_crux_decoder_hf_fidelity_not_latents_20260609.md` (the localization) ·
`b1_clean_pr95_ep1000_verdict_psnr_is_not_d_seg_20260609.md` (PSNR≠d_seg) ·
`deep_hinerv_snerv_fidelity_review_vs_evaluate_py_20260609.md` (HiNeRV manifest, H1-H5/F1-F6) ·
`snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609.md` (cross-vehicle, Mistake A+B).
