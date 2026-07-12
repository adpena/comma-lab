# Frozen SegNet in the witness-training loop: necessity, optimality, alternatives, and P0 builds

Date: 2026-07-12  
Lane: `lane_frozen_segnet_gradient_replacement_elm_headsolve_20260712`  
Axis: `[macOS-MLX training-gradient]` / `[macOS-CPU advisory profiling]`  
Authority: training-loop engineering only. `score_claim=false`; `promotion_eligible=false`. The contest pointer is unchanged; only `upstream/evaluate.py` on exact archive bytes can move it.

## Executive verdict

**Does the stack need frozen SegNet?** At evaluation, yes: `d_seg` is literally the pixel fraction on which the frozen SegNet argmax of the witness differs from the frozen SegNet argmax of the source. There is no substitute authority.

**Does every training step need the full frozen forward?** No. The current zero-model-error relaxation needs a scorer response, but a periodically re-anchored student, exact local response model, or teacher-derived costate can supply intermediate training signals. Full-teacher forward remains mandatory at refresh/verification boundaries and for every score claim.

**Does every training step need the full frozen backward?** No. Backward is not part of the metric definition; it is the present estimator of `dL_relaxed/dx`. It can be replaced by a faithful estimate of the input costate `lambda=dL/dx`, because the renderer parameter gradient is `J_x(theta)^T lambda`. This is the primary structural target.

**Is current use optimal?** The load-bearing kernels are already strong: last-frame-only is exact, fp32 is the measured fast path, and the custom Metal grouped/depthwise backward is default-on and measured at 16.9x with witness-parameter gradient cosine 1.0 after the historical mismatched-init artifact was corrected. But the loop is not globally optimal:

1. the base loss and the surgical raw-margin levers can call SegNet twice on the same composed frame-1; the raw logits can be shared exactly and the loss-only class offset applied afterward;
2. distinct witness-alone frame-1 and temporal frame-0 calls are semantically different and cannot be cached as exact duplicates;
3. activation checkpointing is a memory trade, not a speed win at present;
4. no existing distilled student, #426 costate organ, #36 Atlas, or #141 saliency cache currently replaces the live per-step frame gradient in this trainer;
5. the naive forward-distilled student has already failed descent equivalence despite high argmax agreement, so only gradient-aware/on-policy formulations remain admissible.

**P0 build outcome:** the ELM/INR streaming closed-form affine-SDF-head seed and the fail-closed periodic student/costate/cache contract are built in this landing. The ELM seed is directly consumable by #341 through its existing `--params` surface; it does not claim to solve the non-affine trunk, FiLM, texture head, palette, or exact argmax objective. Its first pair-0 slice is an important scoped negative: the semantic-head proxy improves, but frozen-SegNet disagreement worsens before #341 polish, both before and after R. Therefore the seed is built but **not admitted as a default terminal initializer**.

## Stores consulted / settled lines recalled

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`.
- `.omx/research/fast_witness_training_oss_survey_20260712.md`.
- operator memory `max_throughput_over_bit_identity_operator_override_20260712.md`: training-loop bit identity waived in favor of functional parity and wall time; exact score authority unchanged.
- `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`, especially #36, #141, #341/#342, #426/#428.
- `.omx/research/distillation_sota_survey_20260711T120058Z.md` and `.omx/research/local_throughput_attack_ranked_measured_20260611.md`.
- `.omx/research/amortized_operator_pontryagin_loop_cluster_20260711.md`, `.omx/research/solve_dont_train_inventory_20260709.md`, `.omx/research/basin_finisher_head_solve_probe_measured_20260707.md`.
- `.omx/research/evaluator_response_atlas_20260627.md`, `.omx/research/negative_findings_reaudit_20260710.md`, and the current trainer/scorer sources named below.

Nothing in those four alternative lines was re-derived. The question answered here is narrower: **which line is actually the training gradient provider now, and what would be required to replace it?**

## 1. Necessity verdict: separate the authority-bearing forward from the replaceable backward

### 1.1 What the metric is

`upstream/modules.py::SegNet` is a 9,543,831-parameter `smp.Unet('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)`. Its forward selects `x[:, -1, ...]`, resizes that last frame bilinearly to 384x512, and returns five logits. Its distortion is the mean of:

`1[argmax SegNet(witness_last) != argmax SegNet(source_last)]`.

The exact score therefore requires the frozen forward on the exact realized witness bytes. The source side is immutable, so its exact frozen-SegNet argmax may be computed once and content-addressed; the changing witness side is the forward that must be recomputed for an exact verdict. The hard argmax itself has zero/undefined classical gradient, so training uses differentiable relaxations of the same frozen logits: CE, tau-softplus margin, l7 hard-pixel refinement, or unified `L_tau`. In the base loss, `experiments/train_witness_realized_through_R_mlx.py::make_loss_fn` renders through R and calls `adapter.segnet(f1)` before forming those losses. The level-set trainer differentiates the composed closure with `nn.value_and_grad(model, total_loss_fn)`.

This establishes two different obligations:

| surface | what is necessary | what is replaceable |
|---|---|---|
| exact verdict / score | real frozen SegNet forward, exact R, exact bytes | nothing |
| full-teacher training anchor | real forward on the current candidate; real input gradient when measuring faithfulness | frequency can be reduced if a gate re-anchors |
| intermediate training step | a signal correlated with the current scorer decision geometry | full teacher forward and backward may both be amortized |
| renderer update | `J_x(theta)^T lambda` | how `lambda` is obtained |

### 1.2 Why full backward is sufficient but not necessary

For a relaxed teacher loss `L_T(x)`, the current gradient is:

`dL_T/dtheta = (dx/dtheta)^T (dL_T/dx)`.

Define the pixel costate `lambda_T=dL_T/dx`. If a provider emits `lambda_hat` and training uses the injected scalar

`L_inject(theta)=sum(stopgrad(lambda_hat) * x(theta))`,

then `dL_inject/dtheta=(dx/dtheta)^T lambda_hat`. With `lambda_hat=lambda_T`, this is exactly the full-teacher renderer gradient without keeping or replaying the teacher backward graph. Thus the frozen backward is an implementation choice, not an authority requirement.

The catch is faithfulness: a forward-accurate student need not have the right input Jacobian. The prior c32 student measured 98.8619% teacher argmax agreement and a 12.213x cheaper step, yet exact-teacher descent ended at d_seg 0.308677 versus 0.005797 for full backward. The c64 arm remained non-equivalent: 0.232676 versus 0.005538. Those are formulation-level negatives on logit-MSE distillation, not a family kill; they make gradient matching and on-policy refresh mandatory.

### 1.3 Does it need to be every pair, every step?

No mathematical law requires that cadence. The safe baseline uses it because it has zero model error and automatically follows every parameter update. A replacement cadence is admissible only while all of these stay true:

1. current-frame predicted-vs-teacher costate cosine/norm error passes;
2. a proposed replacement step decreases the real teacher relaxation and has bounded regret versus a real-teacher step;
3. the scorer fingerprint and preprocessing/R chain match;
4. staleness and frame displacement remain inside an explicitly measured trust radius;
5. any failure falls back to the full teacher and forces refresh;
6. exact scorer verification remains periodic and exact archive evaluation remains unchanged.

That is the contract implemented by the prototype in §4.2. No current #426/#247 controller row satisfies it because those lambdas live over campaign state and lever features, not the dense current-frame pixel field.

## 2. Optimality audit of current use

### 2.1 What is already optimal or settled

- **Last-frame-only:** exact, structural, already implemented by upstream SegNet. Frame 0 has zero d_seg gradient unless a separate temporal consistency lever intentionally scores it.
- **fp32:** measured faster than fp16/bf16 on the relevant Apple/arm paths; half precision was also less faithful. Keep fp32 for the teacher path.
- **Custom grouped/depthwise backward:** default-on when Metal is usable. Canonical B8 anchor: 396 ms with the custom path versus 6,713 ms reference, 16.9x. Earlier isolated backward A/B was 20,206 to 570 ms (35.45x); full-scorer backward B4 was 11,149 to 621 ms (17.96x). The four strided depthwise layers were the old wall, and the custom gradient's measured cosine to reference was 0.99999775 at the scorer surface; seeded end-to-end witness-parameter cosine is 1.0.
- **Static target caching:** GT argmax, margins, class maps, skeletons, and fixed transforms are already cached or cacheable. They should remain outside per-step teacher work.
- **Shared surgical-margin forward:** lane, margin-saliency, thin-lane, subpixel, chroma, phase, satisfice, and related levers already share one raw-logit/margin call rather than each making its own call.

### 2.2 Current duplicate-call finding

The sharing boundary stops one level too early. `make_loss_fn` always computes the base-loss frame-1 logits. `total_loss_fn` then renders the same composed frame-1 again and calls raw `adapter.segnet` when a surgical raw-margin lever is active. The reason for the split is valid—the base may use `_LogitAdjustSegAdapter`, while the surgical levers need raw logits—but the second convolution is not required. Compute raw logits once, derive adjusted base logits as `raw + offset`, and pass the raw tensor to all surgical consumers.

This is an exact common-subexpression elimination, subject to a parity test that the two render paths are the same object/value for the composed route. It does **not** eliminate:

- the witness-alone frame-1 call when seed-excluded island formation is active (different pixels);
- the frame-0 call for temporal screw consistency (different frame);
- any deliberately different augmentation/R surface.

Depending on active gates, the current serial step therefore pays the base frame-1 call plus zero or more of: duplicate raw composed frame-1, witness-alone frame-1, and temporal frame-0. The exact count is schedule-dependent. This is the highest-confidence residual exact optimization found by this audit. It is recorded rather than hot-patched because the live trainer is sister-owned in the current work wave.

### 2.3 Measured hot-path context

The canonical stripped MLX seg-only B8 profile with grouped backward ON measured:

| component | ms/step | share |
|---|---:|---:|
| SegNet forward + backward | 399.1 | 69% |
| INR trunk render | 150.2 | 26% |
| R operator | 27.9 | 5% |
| total | 577.2 | 100% |

This denominator is a stripped closure, not the full current V9 schedule. The survey's “~95% SegNet + trunk” is the sum of the first two rows; it must not be read as “SegNet alone is 95%.” A separate older full-stack CPU profile measured scorer fwd+bwd at 98.15% of its own closure. Those measurements have different axes and closures and are not interchangeable.

On that same stripped MLX denominator only, an ideal zero-cost cache that pays the 399.1 ms teacher slice once every `k` steps has algebraic whole-step ceilings of 1.528399x at `k=2`, 2.077193x at `k=4`, and 2.531718x at `k=8`: `577.2 / (178.1 + 399.1/k)`. These are **derived upper bounds, not expected wins**; provider inference/injection, refresh validation, fallbacks, and the full V9 lever schedule all reduce them. They identify why a passing refresh cadence is worth pursuing without fabricating a launch result.

### 2.4 New layer-wise frozen-SegNet profile

Reproducible executable: `tools/profile_segnet_blocks.py`; focused test: `src/tac/tests/test_profile_segnet_blocks.py`. New receipt: `experiments/results/segnet_block_profile_20260712T151901Z/profile.json`, SHA-256 `0968bad12856e71bc665a0e75e481b7a6a29d39f6a2c2286068c9e1892e2054b`. Tool SHA-256: `133a83fd3f536071c5dcba1c0f0d0e465ba6474c3ba28456a9e6acb33506a7bb`; ten focused tests passed in three clean passes, with ruff and `py_compile` green.

Scope: one real pair-0 last frame, arm64 macOS CPU, torch 2.12.1, fp32, six threads, no MKLDNN/MPS/CUDA, two warmups and five measured passes. The receipt content-addresses the 37,545,489-byte input video (SHA-256 `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`), both decoded RGB frames, canonical resized input, SegNet weights, executable, `upstream/modules.py`, `upstream/frame_utils.py`, `pyproject.toml`, `uv.lock`, git state, dependency versions, exact argv, and every input gradient. The input-gradient exercise used mean-squared logits only to traverse the exact frozen graph; it is a compute receipt, not a loss or score verdict. Hooks add overhead, so relative attribution is stronger than absolute time.

Measured medians: forward 1,443.995833 ms; input-gradient backward 443.975417 ms; paired total 1,887.971250 ms. Across the five paired samples, median shares were 76.9675% forward and 23.0325% backward. The raw total range was 1,744.036375 to 2,025.286125 ms, which is why no single-sample absolute timing is treated as authoritative and this fresh run is not compared as a speed regression against the earlier instrumented receipt. Saved-tensor accounting was invariant across all five samples: 728,376,176 logical bytes and 691,354,736 globally unique-storage bytes under the receipt's deterministic first-owner rule; neither is peak RSS.

The block table independently median-aggregates forward time, backward time, and paired block-total share. Consequently a displayed forward median plus backward median need not equal the independently paired total underlying the share; every raw paired value remains in the receipt.

| block | fwd ms | input-bwd ms | fwd+bwd share | first-owner unique MiB |
|---|---:|---:|---:|---:|
| encoder stem | 2.682 | 2.674 | 0.21% | 2.25 |
| encoder post-stem `bn1` | 1.319 | 1.199 | 0.13% | 12.00 |
| encoder block 0 | 22.482 | 25.717 | 2.54% | 51.01 |
| encoder block 1 | 80.954 | 75.853 | 8.51% | 175.23 |
| encoder block 2 | 132.796 | 51.486 | 9.86% | 78.59 |
| encoder block 3 | 151.469 | 22.958 | 9.19% | 49.98 |
| encoder block 4 | 226.687 | 47.216 | 14.52% | 61.17 |
| encoder block 5 | 412.274 | 43.864 | 24.24% | 47.45 |
| encoder block 6 | 242.517 | 15.505 | 13.81% | 28.63 |
| decoder block 0 | 11.604 | 7.059 | 1.00% | 10.79 |
| decoder block 1 | 19.179 | 13.278 | 1.76% | 11.46 |
| decoder block 2 | 23.030 | 20.700 | 2.47% | 19.60 |
| decoder block 3 | 39.065 | 41.364 | 4.18% | 39.12 |
| decoder block 4 | 43.214 | 61.473 | 5.38% | 72.03 |
| segmentation head | 15.080 | 9.338 | 1.17% | 0.00 |

The selected non-overlapping block timings cover a paired-sample median 99.7391% of the hook-instrumented end-to-end total. The median residual is 5.097750 ms (2.269832 ms forward and 2.968961 ms backward); it is defined as same-sample end-to-end time minus the sum of disjoint selected-module hook times, not as a separately timed module. Adding the post-stem encoder `bn1` closes saved-tensor attribution completely: unattributed logical and unique-storage bytes are both zero in all five samples.

Encoder blocks 5 and 6 consume a paired-sample median 37.6620% of the measured total but own only 76.08 MiB of globally unique saved storage combined. Encoder block 1 is the activation-memory outlier (175.23 MiB), not the time outlier. This separates the remedies: late encoder optimization is a compute question; early-block checkpointing is a memory question.

There is a second consequence: on this CPU axis, deleting **only** the backward while retaining the full teacher forward has a paired-sample median scorer-slice ceiling of 1.299250197x, before any rest-of-loop cost. The larger structural target is therefore not merely a cheaper VJP. A cached costate skips teacher forward and backward between refreshes; a passing student replaces both with its smaller forward/backward. This distinction matters even more after the Metal grouped backward has already removed the pathological depthwise VJPs.

### 2.5 Audit of the previously un-questioned choices

| choice | verdict | reason / next gate |
|---|---|---|
| truncate / early-exit backward | **not exact** | Final logits depend on every encoder/decoder block. Freezing weights does not remove the need to propagate input gradients through them. An auxiliary early-exit head is a learned surrogate and must pass the same costate/descent gate. |
| lower-resolution gradient path | **proxy only** | SegNet already defines its function at 384x512 after the canonical resize. Any further downsample changes receptive-field/boundary behavior. Test only as an on-policy student/cache arm, with real-teacher one-step and short-horizon verification. |
| activation checkpointing | **throughput-negative unless it unlocks batching** | Checkpointing encoder blocks 1/2 plus decoder block 4 targets 337.855 MiB logical, 48.6379% of saved tensors, but the paired-sample median forward replay is 250.206416 ms. On memory-rich local runs this is a loss. Use only if the released memory enables a measured larger scorer batch. |
| activation caching | **static yes, dynamic no** | Source logits/labels/margins and fixed maps are cacheable. Candidate encoder activations are theta-dependent; exact cross-step reuse is invalid. A local response cache is an approximation and needs a trust radius/refresh gate. |
| grouped backward maximality | **near the safe local optimum, not a proof of global maximality** | It already removes the four pathological depthwise VJPs. Tiling/coalescing or fusing grad-input/grad-weight has only a prior labeled estimate of 1.1–1.3x on the SegNet slice and high correctness risk. Revisit only behind full parity and isolated benchmark gates. |
| micro-batch / batched twin | **live sister gate pending** | Prior K batching was device-dependent and sometimes negative; the current training-loop bit-identity waiver reopens functional batching. Existing K=8 scorer-forward anchors are 1.56x GPU / 1.75x CPU, but end-to-end V9 functional parity and wall time are still owed by the sister lane. Do not count an unmeasured n600 win here. |
| full scorer every step | **safe baseline, not necessary** | Replace only through the policy in §4.2; refresh failure falls back to full teacher. |

## 3. Recalled alternatives, current wiring truth, and ranking

Ranking uses the requested product qualitatively: wall-clock ceiling x implementation feasibility x witness faithfulness. No numerical composite is invented. “Rank” is a build/admission order under current evidence, not a family kill.

| rank | line | current reality | wall-clock / feasibility / faithfulness | decision |
|---:|---|---|---|---|
| 1 | **Periodic real-teacher local response / Jacobian-costate cache** (#36 Atlas + #141 input-margin Jacobian) | Atlas and saliency producers exist; the trainer consumes cached `S_R` only as a stop-gradient weight while still running live SegNet. No cache supplies the training gradient. | High if a refresh interval >1 survives; medium-high feasibility; exact at refresh and locally faithful only inside a measured radius. | **Top structural prototype:** exact teacher costate at refresh, injected between refreshes; measure current-frame costate agreement and teacher-step regret before admitting any cadence. |
| 2 | **On-policy Sobolev/Jacobian-distilled student** (#428 distillation line) | Tiny Torch/MLX student surfaces exist, but none is the current level-set gradient provider. Naive logit-MSE student is measured non-equivalent despite 12.2x step speed. | Largest measured compute ceiling; medium feasibility; faithfulness currently failed for the naive formulation. | **Reactivate only** with input-Jacobian/random-projection Sobolev loss, boundary-annulus weighting, on-trajectory replay, and periodic full-teacher refresh. A passing student can replace both fwd+bwd between anchors. |
| 3 | **Dense costate-amortized gradient** (`lambda_hat(frame,state)`) | #426/#247 predicts campaign-level lambda over d_seg-by-class/log-bytes and lever features. It does **not** predict the 384x512x3 frame costate and is advisory/control-only. | Very high backward-removal ceiling; medium-low feasibility because the output field is dense; faithfulness unmeasured. | Build after the injection/gate contract: learn a low-rank/annulus factored lambda field, not a full unconstrained image, then use periodic true costates as labels. |
| 4 | **Solve-don't-train** (#341/#342/#73/#155), now with an ELM-seed implementation | #341 head chart near-quadratic is measured, but K=8 subset overfit made the post-run formulation +5.1% worse at n600; full-P remains the only admissible form. The new ELM p1 slice also improves its own semantic-head proxy while worsening real frozen-SegNet disagreement. #73/#155 are offline feasibility/codec lines, not current gradient providers. | Modest whole-run ceiling; high implementation feasibility; witness faithfulness is negative for the measured ELM p1 formulation and unmeasured after full-P GN polish. | **Built, not admitted:** retain as a governed proposal seed only. Require full-P #341 exact-loss GN/CG to recover both the baseline and exact scorer debt before use. It removes no main-loop scorer work. |

Why the cache ranks above the student under current evidence: it starts from the exact real-teacher derivative and only models local evolution, whereas the student has a measured basin-divergence negative. Why ELM ranks fourth despite being built first: it is the most concrete to implement, but it cannot remove the main-loop scorer and its first receiver check is negative.

The measured-inert LEVER-4 texture multiplier does **not** falsify a real Jacobian cache. It falsifies the formulation “image texture is a stand-in for through-R reachability”: Pearson -0.033 and top-5% Jaccard 0.024 approximately chance. The correct cache object is the actual `S_R`/costate plus a refresh/trust region, not texture.

A fifth, newly audited line is forward-only finite differences/SPSA. It is **not ranked for build**: on the measured CPU slice, even one two-sided direction needs roughly two 1,443.995833 ms teacher forwards to avoid a 443.975417 ms backward, already exceeding the 1,887.971250 ms full forward+backward median before renderer work, while estimating a 589,824-dimensional boundary field from one direction is high variance. This is a scoped NO-GO on the measured CPU formulation, not a statement about every accelerator or structured low-dimensional terminal chart; #341 is the structured solve that makes the latter viable.

## 4. Builds

### 4.1 ELM/INR streaming affine-head seed for #341

Implementation spec: `.omx/research/frozen_segnet_gradient_elm_implementation_spec_20260712.md`.

Built surfaces:

- `src/tac/boundary_math/elm_inr_head_solve.py`
- `tools/elm_inr_head_seed.py`
- `src/tac/tests/test_elm_inr_head_solve.py`
- `src/tac/witness_dsl/elm_head_seed_policy.py` plus policy test
- `src/tac/canonical_equations/elm_inr_affine_head_seed_20260712.py` plus equation test

The build extends the settled in-memory global helper `fit_out_sdf_to_structured_target`; it does not replace or re-derive that result. It ports the ELM paper's transferable mechanism—freeze `H`, solve the affine output layer, and blend local solves—onto the actual existing directional-feature/level-set hidden state. It does **not** claim to reproduce the paper's random-Fourier architecture: replacing the current hidden basis would change the receiver rather than seed #341. It freezes the actual hidden feature/trunk state, forms label-smoothed centered log-probability targets, accumulates weighted ridge normal equations per rectangular partition-of-unity subdomain, solves local affine heads, and streams a second ridge projection back into the current single global `out_sdf` head. With one subdomain the solve is directly deployable; with multiple subdomains the projection RMSE explicitly reports how much local POU expressivity the current receiver cannot represent. A 1x1, zero-ridge test matches the settled helper.

Only `out_sdf.weight/bias` change. `out_tex`, palette, FiLM, codes, trunk, and all metadata are preserved. Ridge, smoothing, grid, scope, and the three input digests compile through a frozen value-provenanced policy; the CLI has no loose semantic bypass. Every stage invocation recomputes exact SHA-256 over the 5,078,017,610-byte label cache (measured at 9.57 seconds on this host) before resume is accepted. The source labels are range-validated as classes 0..4 and compacted from source `int64` to retained `uint8`, reducing resident label storage from 943,718,400 to 117,964,800 bytes (exactly 8x) without changing labels. The output is a complete atomic checkpoint accepted by:

`.venv/bin/python tools/quadratic_basin_finisher_probe.py solve --params <elm-seed.npz> --tag <feature-state> --mask head --k-pairs 600 ...`

This is a **seed** for the measured near-quadratic #341 chart. It is not a one-shot exact d_seg solve. Full-P=600 GN/CG under the real tau-stage through-R loss remains owed and governed; the K=8 subset is not revived.

**Build/test/bounded-slice receipt:** 30 focused tests passed in three clean passes; ruff and `py_compile` passed. A real CLI-level interruption/resume regression compares an uninterrupted three-invocation run with an interrupted/resumed five-invocation run: both final direct and POU checkpoints are byte-identical, and done-state re-entry revalidates final custody. The pair-0 v3-schema receipt is `experiments/results/elm_inr_head_seed_20260712/faithful_slice_r2/elm_head_seed_receipt_diagnostic_p1_g2x2.json`, SHA-256 `3d5e7571892c145cda2362d63a7e94537c811b422061b7743b2838e0e5c46aa0`; the complete v4 resume state `elm_head_seed_state_diagnostic_p1_g2x2_v4.npz` has SHA-256 `f7b08fd2625b659be68a499579896fd02dc7ed1dcdfc7197c11d73a9bf104fdf`. Both 458,622-byte outputs preserve all 51 non-head arrays exactly. The direct-global target-SSE optimum has checkpoint SHA-256 `544311911ed4567413704819f1f9b9892cf157c8fde04848b0e5b2c1d40a347e`; the POU-fold checkpoint has SHA-256 `3a5cfd45c2de387b703c2cabedaf07e7cd14a64b4593d12290371a798ba2b44b`. Diagnostic scope emits no #341 command. A full-P policy can emit a suggested command only when the exact canonical feature-state tag resolves to the content-addressed feature-state path and its bytes still match the declared SHA when rehashed immediately before command emission; path/tag equality alone is insufficient. Otherwise it records a fail-closed custody blocker. Every emitted command spells `--k-pairs 600` explicitly.

The four receiver metrics expose the key structural result: direct-global target RMSE was 0.2191140756327058, local POU target RMSE was 0.19337069097479676, folded-global target RMSE returned to 0.2191140756327058, and fold-vs-local RMSE was 0.04642787892207622. All four local systems, the direct comparator, and the global fold were rank 97/97. The local field improves its proxy only before it is projected into the receiver that actually ships; after folding it cannot beat direct global LS on the same target-SSE objective.

The realized-through-R receiver check is negative and controls interpretation of those LS numbers. The standalone comparison receipt `experiments/results/elm_inr_head_seed_20260712/faithful_slice_r2/through_r_pair0_comparison.json`, SHA-256 `1756e0d6392720fb6950a3405fde5c3fbdb68c4a7dbf1a5ee2a45b12d10a6737`, binds the exact command/cwd, scorer weights, GT cache, source video, feature state, policy, all three checkpoints, implementation/upstream/R sources, dependency locks/versions, runtime axis, and timing boundaries. Its typed pair-0 command remeasured 685/196,608 disagreements, or 0.0034840901692708335, for source checkpoint SHA `6dd28a6e295d007ef0e53ae3e0e792a517a5708394a17d2185870e44920dedca`. Both the direct-global and POU-fold seeds measured 1,367/196,608, or 0.006952921549479167: +0.0034688313802083335 absolute, +99.56204379562044%, or 682 additional disagreeing pixels. The entire comparison reached receipt-payload preparation in 9.714447 seconds under its 58-second guard; one batch-of-three frozen-SegNet measurement took 1.072770 seconds, and no per-candidate scorer time is inferred. This is a formulation-level **NO-GO for the pair-0, smoothing=0.1, temperature=0.30982047258105083, 2x2-POU, ridge=0 seed as a standalone initializer**, not an ELM-family kill. No Gauss-Newton, d_pose, archive, n600, or score was evaluated. Full-P=600 plus #341 exact-loss recovery remains governed and owed.

### 4.2 Fail-closed in-loop gradient replacement contract

Implementation spec: `.omx/research/frozen_segnet_gradient_replacement_contract_spec_20260712.md`.

Built surfaces:

- `src/tac/boundary_math/segnet_gradient_replacement.py`
- `src/tac/witness_dsl/scorer_gradient_policy.py`
- `src/tac/canonical_equations/segnet_costate_injection_20260712.py`
- behavioral tests and `tools/probe_segnet_costate_injection.py`

The contract provides Torch and lazy-MLX costate injection, mandatory global gradient-fidelity metrics, optional additional annulus metrics, explicit typed refresh/trust thresholds, changed-fingerprint/age/nonfinite fail-closed behavior, and a one-step real-teacher regret gate. Objective context binds scorer, preprocessing, R, GT, pair, loss, stage, and parameters; the compiled fingerprint is re-derived on every decision so post-compile dictionary mutation fails closed. Observations/checks additionally bind exact frame content, provider custody, and step, preventing mask laundering, cross-frame reuse, cross-objective reuse, and replay. At a teacher refresh, the anchor and current frame hashes must be equal and the teacher-validated anchor provider costate must be byte-identical to the current costate that will actually be injected. Provider bytes are fully hashed at compile/refresh; between refreshes a cheap device/inode/size/mtime/ctime fingerprint detects mutation and forces fallback. Modes distinguish full teacher, periodic student, periodic dense costate, and trusted Jacobian cache. The module is intentionally not wired into the live trainer while sister agents own that hot file.

The synthetic proof verifies the chain-rule identity through a nontrivial renderer. The optional real-SegNet slice exercises short-horizon endpoint behavior and teacher-call cadence at refresh intervals 1/2/4 without an n600 launch; it does not by itself measure fresh-versus-cached costate staleness at each intermediate step.

**Build/test/prototype receipt:** 36 focused tests passed in three clean passes; ruff and `py_compile` passed. The synthetic behavioral receipt `.omx/research/artifacts/segnet_costate_injection_probe_20260712.json` (SHA-256 `add14a28d74dd71d5ab6bc51673e4d6b9a5f2f11eb73bc42bb4119ec6e5524ea`) measured direct-versus-exact-injected renderer-parameter gradient cosine 1.0, relative L2 0.0, and max absolute error 0.0; the sign-reversed negative control measured cosine -1.0 and relative L2 2.0.

The optional real frozen-SegNet n=1 slice completed under its 58-second guard in 20.73 seconds; receipt `.omx/research/artifacts/segnet_costate_cache_slice_n1_20260712.json`, SHA-256 `f59193945df680be9782e139c7d1664ca7a1fe58a40fa6cb8137a01efde15424`. Over four low-dimensional renderer steps, refresh intervals 1/2/4 used respectively 4/2/1 training teacher forwards and backwards. All three ended at exact-teacher n=1 d_seg 0.0029652912635356188 from 0.00299072265625; total per-arm wall was 8.6700/4.9446/3.0627 seconds. This is descriptive direct-slice **endpoint/cadence** evidence only: one pair, one CE regime, twelve renderer parameters, shared initialization, and a final teacher check. `fresh_nonrefresh_costate_agreement_measured=false`; it does not establish intermediate cached-gradient faithfulness, a generalized wall-clock win, or a safe refresh interval for the live trainer.

## 5. Concrete next build recommendation

### 5.1 Immediate exact optimization before approximation

At the next safe trainer edit window, eliminate the duplicate same-frame f1 SegNet call:

1. render composed f1 once;
2. compute raw logits once;
3. form adjusted base logits by adding the live class offset;
4. pass raw logits/margin to surgical levers;
5. retain separate witness-alone and frame0 calls only where their inputs genuinely differ;
6. prove loss and renderer-parameter gradient functional parity on the full active V9 lever set, then measure end-to-end wall time.

This is higher confidence than truncated/coarse approximations because it changes no function.

### 5.2 Structural arm to build and fire

Build the **periodic on-policy Sobolev student** first, using the contract landed here:

- student input: current realized frame after the same R/preprocess; the unchanged teacher loss still consumes the exact GT map and current stage parameters, so the student replaces scorer logits/Jacobian rather than inventing a target-free loss;
- targets at teacher refresh: full logits/features plus real input costate projections;
- loss: logit KD + boundary-annulus decision loss + random-projection Jacobian/Sobolev matching; no forward-agreement-only admission;
- replay: recent on-trajectory frames across CE, tau, l7, and terminal regimes;
- runtime: student supplies fwd+bwd between anchors; every `k` steps full teacher refreshes and verifies current-frame costate metrics plus one-step teacher loss;
- governor: any gate failure, scorer/hash drift, or trust-radius exit falls back to full teacher and resets the cache/student anchor;
- measurement: report end-to-end wall, teacher duty cycle, exact-teacher short-horizon descent, and full-P governed A/B. Do not infer a win from the historical 12.2x isolated student step.

In parallel, use the same teacher gradients to train a low-rank annulus costate arm. Condition it on the content-addressed GT/pair/loss/stage objective context as well as the current realized frame. The current #426 organ can provide regime/context features, but it is an input to the new predictor, not the predictor itself. Factor the dense field as boundary basis coefficients plus a small bulk residual so the model learns the scorer-active annulus rather than 589,824 unconstrained pixels.

### 5.3 Admission sequence

1. exact duplicate-f1 common-subexpression elimination;
2. n=1/short-slice exact-costate cache radius measurement;
3. on-policy Sobolev student versus cache versus full teacher on identical slices;
4. bounded n24/n96 multi-regime functional A/B with periodic exact teacher;
5. governed n600 training A/B, resumable and per-stage checkpointed;
6. byte-close and `upstream/evaluate.py` on exact bytes for any score claim.

## 6. Triality and pointer delta

- **DAG leg:** this memo plus the ELM, gradient-replacement, and block-profiler implementation specs; they connect #36/#141, #341/#342, and #426/#428 to the actual training-gradient seam.
- **DSL leg:** typed `elm_head_seed_policy` compiles the closed-form solve without loose semantic flags; typed `scorer_gradient_policy` compiles replacement modes and fails closed on missing custody/refresh/trust fields. No invented live-trainer flag is emitted.
- **Equations leg:** `elm_inr_affine_head_seed_20260712` and `segnet_costate_injection_20260712` encode the closed-form normal equations and the exact chain-rule injection identity.
- **Durable artifacts:** code, tests, profiler JSON, prototype receipt(s), and this memo.

Final focused seal: 76 tests passed in three consecutive clean combined runs; ruff and `py_compile` passed across all 16 landed Python files. Independent adversarial review closed the feature-state command-emission, comparison-receipt, profiler coverage, and gradient-refresh custody blockers before serialization.

Pointer delta: **none**. No archive was built or evaluated, no n600 run was launched, and no score moved. Owed heavy work remains behind the governed launcher.
