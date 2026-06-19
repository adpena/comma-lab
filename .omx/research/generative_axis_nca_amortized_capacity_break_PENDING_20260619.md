---
title: "GENERATIVE-AXIS FINAL EXHAUSTION TEST — amortized continuous-texture NCA + capacity-break sweep (PENDING daemon)"
authority: "[contest-CPU advisory] NON-PROMOTABLE — pointer UNMOVED 0.19110; $0; MPS-gradient/CPU-authority; no PR"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-19
verdict: PENDING_DAEMON_SWEEP
superseded_by: .omx/research/generative_axis_nca_amortized_capacity_break_RED_20260619.md
producer: experiments/probe_nca_texture_amortized_capacity_break.py
daemon_log: .omx/tmp/nca_daemon/main.log
state: experiments/results/nca_amortized_capacity_break_main/gate_state.json
cross_refs:
  - .omx/research/generative_axis_continuous_texture_nca_AMBER_20260619T020000Z.md   # the AMBER this fixes
  - .omx/research/dseg_side_feasibility_corners_verdict_20260619.md                   # the d_seg wall (frontier 0.00056)
  - .omx/research/p_suff_task_ablation_verdict_20260619.md                            # frontier near task-RD floor
  - .omx/research/generative_axis_dseg_core_design_20260619T004600Z.md                # the 29.3*params^-0.71 wall
---

# Generative-axis FINAL exhaustion test — PENDING (this skeleton is replaced on daemon completion)

The LAST un-run path of the sub-0.15 campaign: a best-shot AMORTIZED continuous-texture NCA decoder,
fixing the AMBER's three caveats and answering the original hypothesis (does weight-shared ITERATION break
the `d_seg ~ 29.3·params^−0.71` capacity wall?). Daemon running; this memo is filled on completion.

## The three caveats this fixes (the build mandate)
1. CONVERGENCE FRAGILITY (~2/8 AMBER runs converged) -> Mordvintsev POOL + sample-replay + multi-restart
   keep-best + per-param grad-norm + LR warmup + step-decay. MPS is the only tractable gradient device
   (CPU-gradient measured ~0.7s/it single-frame = impractical for the sweep); multi-restart keep-best is
   the convergence-robustness mechanism on the available hardware.
2. AMORTIZATION UNTESTED (n=1 per rule) -> ONE shared rule across 8 REAL GT frames + per-frame latents;
   AVERAGE d_seg (NOT best-frame) + TRUE amortized rate (rule once / 600 + latent × 600).
3. CAPACITY-BREAK NEVER SWEPT -> rule-size sweep (C8h32 ~10k → C32h256 ~70k params, an ~7× span); fit the
   AVERAGE d_seg(params) exponent vs the power-law k=0.71.

## Convergence-engineering journey (caveat 1) — the measured findings so far (NO-FAKE)

The build's FIRST hard blocker was convergence-robustness, exactly as the AMBER predicted. The measured
sequence (all `[contest-CPU advisory]`):

| config | iters | device | stabilizers | result | finding |
|---|---|---|---|---|---|
| amortized 4f batch2 | 600 | MPS | grad-norm+warmup+fire0.5+pool0.5 | collapse 0.508, recon 110, 0/4 | under-trained + fire/pool noise |
| single-frame | 800 | MPS | grad-norm+warmup, no-fire | partial 0.264, recon 76, 0/1 | under-trained vs AMBER's 1500it |
| amortized 8f batch2 | 2400 | MPS | grad-norm+warmup+pool0.3 | **NaN** (recon=nan), 0/8 | **pool feedback -> unbounded state growth -> inf** |
| amortized 4f batch2 | 400 | MPS | grad-norm+warmup, **state-bound 32, no-pool** | partial 0.36, **recon 45.6 FINITE**, descending | **NaN FIXED** (tanh state-bound = alive-masking surrogate) |

**Two real convergence findings:**
1. The Mordvintsev POOL + sample-replay, applied to this objective (texture regression through a frozen
   scorer with a trained residual rule), is DESTABILIZING, not stabilizing — the pool feeds grown states
   back which grow unboundedly through the deep unroll to inf/NaN. (Mordvintsev avoids this with
   alive-masking, which bounds growth; the RGBA-emoji morphogenesis target is different from a fixed
   texture.) The fix is a soft tanh STATE-BOUND (the alive-masking surrogate) + dropping the pool.
2. CPU-gradient (the deterministic dodge for MPS non-determinism) is ~0.7s/it single-frame =
   impractically slow for the sweep; multi-restart keep-best on MPS is the tractable convergence-robustness
   mechanism on the available hardware.

The fixed recipe (state-bound + no-pool + no-fire + grad-norm + warmup + step-decay + multi-restart) is
numerically stable (no NaN); the capacity-break sweep runs on it.

## The fork (filled on completion)
- GREEN: byte-closed S < frontier 0.19110 (ideally sub-0.15).
- AMBER: reliable convergence + shared-rule holds d_seg, S in [0.15, 0.19).
- RED: average d_seg(params) obeys ~^−0.71 -> generative axis caps like the rest -> the FINAL family.

(Numbers + final verdict pending daemon completion; analysis via experiments/analyze_nca_capacity_break.py.)
