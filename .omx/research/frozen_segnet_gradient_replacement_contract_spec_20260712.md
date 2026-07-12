# Frozen-SegNet P0: in-training gradient replacement contract spec

Date: 2026-07-12  
Lane: `lane_frozen_segnet_gradient_replacement_elm_headsolve_20260712`  
Scope: build a fail-closed, trainer-independent prototype; no live-trainer or V9-arm edit.

## Why this is a new contract, not a replay of the old student

The existing direct-pixel A/B is already decisive for the naive formulation: a c32 logit-MSE student was measured 12.213x cheaper per step and reached 0.988619 teacher argmax agreement, but its exact-teacher d_seg after 80 descent steps was 0.308677 versus 0.005797 for the full scorer gradient (`surrogate_descent_equivalence_c32_v2.json`). The c64 arm remained non-equivalent (0.232676 versus 0.005538). Therefore forward/logit agreement is not an admission gate. The missing object is the input costate `lambda = dL_teacher/dx`.

## Owned files

- `src/tac/boundary_math/segnet_gradient_replacement.py`
- `src/tac/witness_dsl/scorer_gradient_policy.py`
- `src/tac/tests/test_segnet_gradient_replacement.py`
- `src/tac/tests/test_scorer_gradient_policy.py`
- `src/tac/canonical_equations/segnet_costate_injection_20260712.py`
- `src/tac/canonical_equations/tests/test_segnet_costate_injection_20260712.py`
- `tools/probe_segnet_costate_injection.py`

Do not edit `curriculum_dsl.py`, the live trainers, CUDA, micro-batch modules, or the live V9 result tree.

## Core identity

Let the renderer produce frame `x(theta)`, and let a frozen/stop-gradient provider supply `lambda_hat` with the same shape as `x`. Define:

`L_inject(theta) = sum_j stopgrad(lambda_hat_j) * x_j(theta)`.

Then `dL_inject/dtheta = J_x(theta)^T lambda_hat`. If `lambda_hat = dL_teacher/dx`, this is exactly the parameter gradient that the full teacher backward would have supplied, without retaining or replaying the teacher graph. The prototype must demonstrate this identity with a nontrivial differentiable renderer and teacher, including a negative control with a wrong costate.

Provide Torch and lazy MLX implementations. MLX import or device absence must not break NumPy/Torch consumers.

## Faithfulness metrics and gate

Pure NumPy metrics compare `lambda_hat` against a periodically measured real-teacher input gradient on the current rendered frame:

- cosine similarity;
- relative L2 error;
- norm ratio;
- finite-value and shape checks;
- optional mask-restricted metrics for the scorer-margin annulus.

Admission also requires a one-step teacher check: applying the proposed parameter/frame step must reduce the real teacher relaxation and must satisfy an explicit maximum loss-regret relative to the real-teacher-gradient step. Every threshold is explicit in the typed policy; there are no permissive defaults. A missing teacher observation, stale observation, changed scorer fingerprint, nonfinite value, or failed metric causes fallback to full teacher.

## Typed policy / DSL leg

Modes:

1. `full_teacher`: current baseline.
2. `periodic_student`: a small differentiable student supplies fwd+bwd between real-teacher refreshes.
3. `periodic_costate`: a learned lambda network supplies the pixel costate and the injection identity supplies renderer gradients.
4. `trusted_jacobian_cache`: a refreshed local Jacobian/costate cache, valid only within an explicit frame trust radius.

Replacement modes require: refresh interval, maximum staleness, scorer fingerprint, minimum cosine, maximum relative L2, norm-ratio band, maximum teacher-loss regret, and checkpoint/cache custody. Compilation must reject absent/invalid fields. This policy is a contract surface only; it does not invent trainer flags or actuate the live run.

## Prototype receipt

`tools/probe_segnet_costate_injection.py` writes a durable JSON receipt containing seed, framework/version, shapes, direct-vs-injected parameter-gradient cosine and max absolute error, negative-control metrics, and `score_claim=false`. It performs no scorer evaluation and makes no throughput or d_seg claim.

An optional bounded `--real-segnet-cache-slice` mode may assess the nearest concrete cache formulation: compute the exact frozen-SegNet input costate on one real last frame, inject it through a low-dimensional differentiable renderer, and compare deterministic short-horizon refresh intervals 1/2/4. Record exact-teacher final CE/d_seg, teacher forward/backward call counts, and measured wall time. This is an n=1 direct-slice diagnostic only; it may establish a local trust-radius signal, never an n600 or current-trainer win. If a faithful bounded run cannot be completed, emit the blocker rather than synthesizing a result.

## Promotion gate for a future trainer integration

A future trainer patch may select a replacement only after a governed faithful-slice experiment records:

1. on-trajectory real-teacher gradient metrics at multiple training regimes;
2. one-step and short-horizon exact-teacher loss descent versus full backward;
3. measured teacher-refresh duty cycle and end-to-end wall time, not model FLOPs;
4. full-P=600 governed A/B with periodic exact scorer re-verification;
5. fallback activation and resume state preserved at every stage.

The recommended first learner is an on-policy Sobolev/Jacobian-distilled small student because it can replace both teacher forward and backward between refreshes. Direct costate prediction is the next arm: it targets the expensive backward more precisely but must emit a dense 384x512x3 field and is not the campaign-level #426 lambda network already in tree.
