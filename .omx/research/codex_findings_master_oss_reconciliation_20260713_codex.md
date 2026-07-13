---
title: Master OSS reconciliation of four frozen-SegNet throughput probes
date_utc: 2026-07-13T01:46:43Z
lane_id: lane_449_master_oss_reconciliation_20260713
review_status: fresh-eyes-reviewed(3)-CLEAN
score_claim: false
pointer_moved: false
research_only: true
---

# Master OSS reconciliation: frozen-SegNet task 449

## Outcome first

**DERIVED · fresh-eyes-reviewed(3)-CLEAN:** none of YOPO, INSTANT, SFESS, or JRD clears the task-449
frozen-SegNet throughput bar in the measured scope. YOPO has near-unity measured non-refresh costate cosines but
loses its exact admission/economics gate to validation cost. INSTANT admits no arm after the shared typed cycle
law charges measured projected-candidate validation, and it retains the exact forward. SFESS adds the paper's learned-logit optimization and sample
ladder without changing the same-budget result, while a live use would add exact objective forwards. JRD is an
archive-rate postprocessor and removes no scorer call. The surviving direction is therefore the separately scoped
amortized nonlinear surrogate that replaces the frozen forward and is trained on-policy. That direction is
DERIVED, currently unmeasured and unpromoted; this pass did not build or measure it.

**MEASURED · inherited receipt custody:** `reports/latest.md` remained read-only. The YOPO and SHARE_GE2
receipts both record the defensive `[contest-CPU]` pointer as `0.1880443979880752`. Every reconciliation receipt records
`score_claim=false` and `pointer_moved=false`. No paid dispatch, live-trainer edit, or live-run-directory mutation
was performed.

## STORES CONSULTED

STORES CONSULTED: one `tools/corpus_query.py` query loaded research (5715), equations (622), memory (1893),
DAG (505), council (277), tasks (96), and docs (92); the operating manual; the corrected goldmine memo; the
frozen-SegNet costate contract; all four landed receipts, implementations, tests, task rows, equation rows, and
DAG feeds; the SHARE_GE2 memo and sealed diagnostic receipt; the official repository and paper surfaces listed
below. Deliberately not loaded or actuated: the protected live V9 run, paid/cloud providers, the live trainer,
`upstream/evaluate.py`, source-video expansion, or any contest scorer run.

## SHARE_GE2 gate and validation economics

**UNKNOWN · fresh-eyes-reviewed(1) at source receipt:** `share_{>=2}(tau)` is not identified at early,
boundary, or late state. The sealed checkpoints expose `__cfg_softmax_temp` values
`0.8062483931715706`, `0.21682465292832676`, and `0.2156894834900186`, but no verified definition maps
that field to the formula's `tau`, and no definition maps an operator-valued SegNet layer Jacobian to scalar
`beta_i`. The requested O(L) forwards were correctly skipped: they would create numbers without making either
mapping valid. No supporting paper for the operator-supplied formula was identified. The gate is non-load-bearing;
the empirical exact-teacher descent/regret gate decides linear fidelity instead.

**MEASURED · inherited YOPO receipt fresh-eyes-reviewed(2):** the companion law
`K*t_exact/(t_exact+(K-1)*(t_approx+t_validate+t_fallback))` is load-bearing. Its measured non-refresh ratios
were early K2 `0.08127247006954162`, boundary K2 `0.08592634915652758`, late K2
`0.117047659443115`, and late K4 `0.05863237060084415`; early and boundary K4 are UNKNOWN because
no non-refresh step was recorded. Thus linear fidelity is necessary but not sufficient: validation economics
closes the measured YOPO formulation.

## YOPO reconciliation

| Ours, clean-room | Official reference | What the conservative pass missed or could not import |
|---|---|---|
| Bank `p1=dL/dz1` at the frozen first-block cut; recompute `J_prefix(x_t)^T p1`; fail closed on custody, descent, or through-R non-worsening; test K={1,2,4}. | Dinghuai Zhang, Tianyuan Zhang, Yiping Lu, Zhanxing Zhu, and Bin Dong · 2019 · *You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle* · arXiv:1905.00877. The official repository is `a1600012888/YOPO-You-Only-Propagate-Once`. | The public root shows the first-layer/maximal-principle training structure, but no repository-level LICENSE is visible and the required clone failed DNS. No bytes were copied. The source-level banking cadence and refresh implementation remain UNKNOWN. |
| Exact validation is charged at every reused step. | Jacob H. Seidman, Mahyar Fazlyab, Victor M. Preciado, and George J. Pappas · 2020 · *Robust Deep Learning as Optimal Control: Insights and Convergence Guarantees* · arXiv:2005.00616. | The follow-up supplies convergence analysis via inexact-oracle methods, not a cheaper exact through-R validation primitive. It does not erase the measured validation charge. |

**MEASURED · inherited clean-room receipt fresh-eyes-reviewed(2):** 48 teacher forward/backward calls, 402
operational validation forwards, minimum cosines `0.9998774504768612` globally,
`0.9998451425983044` on the boundary annulus, and `0.9999437090802523` at the renderer gradient.
Every complete K>1 cadence was slower than K1; K1-to-arm wall ratios were `0.422978` to `0.683734`.

**UNKNOWN · reconciliation fresh-eyes-reviewed(3)-CLEAN:** no licensed source delta was imported, so no
OSS-enriched rerun exists and the measured delta versus clean-room is UNKNOWN rather than zero.

**VERDICT · inherited formulation fresh-eyes-reviewed(2): NO-GO.** `verdict_scope=n=1 pair0 at sealed
early/boundary/late saved regimes on macOS-CPU advisory; first-block split; K={1,2,4}; event-conditioned
CE-decrease and d_seg-nonworsening fractional recess.` This does not close other splits or the YOPO family.

## INSTANT reconciliation

| Ours, clean-room WIP | MIT OSS reference | What was added |
|---|---|---|
| The dead WIP projected every Conv2d on the spatial axis and never produced a terminal receipt. | Tuan-Kiet Doan, Trung-Hieu Tran, Enzo Tartaglione, Nikola Simidjievski, and Van-Tam Nguyen · 2026 · *INSTANT: Compressing Gradients and Activations for Resource-Efficient Training* · ICLR 2026 · OpenReview:P2q6Y7UweV. The official repository is MIT licensed. | Replaced algorithmically with exact full forward; projection only on eligible ungrouped 1x1 Conv2d input adjoints; adaptive smaller channel/spatial axis; retained-energy rank calibration; oversampling 5; all other adjoints exact. No upstream file bytes were copied. |
| No valid descent/economics tournament. | The official computer-vision path calibrates a low-rank subspace and applies low-rank backward kernels while forward activations propagate normally. | Added three sealed on-policy calibration states, targets {0.90,0.95,0.99}, renderer-gradient positive-cosine admission, exact-teacher CE/d_seg recess, paired timing with median-minus-MAD > 1, positive/negative canaries, atomic stage checkpoints, and terminal no-rewrite resume. The local probe uses three sealed calibration states; it does not claim the official README's `calib_iter=5` setting was reproduced. |

**MEASURED · fresh-eyes-reviewed(3)-CLEAN:** the implementation covers 90 eligible pointwise
layers and keeps 35 other Conv2d layers exact. None of the nine regime/target arms was admitted. The largest
paired hot-step median-minus-MAD ratio was `1.0353914790745997`, so hot-step timing alone would admit some arms. After charging measured
projected-candidate validation under `K={2,4,8}`, the largest optimistic cycle ratio was
`0.5888733451533681`; calibration and fallback were both set to zero as explicit optimistic lower-cost
assumptions. The late-L7 target `0.95` arm retained renderer-gradient cosine `0.4499864310744855`, but its
hot-step lower bound was `0.7506562830313536` and its best charged-cycle ratio was `0.2145066416781345`.
The completed receipt SHA-256 is `09dbd49c8410c23f2e312670ff517c873e793cde7e64a88fb2df3768586b0443`;
an externally hash-bound terminal resume preserved the receipt, calibration checkpoint, and all three stage files.
The rerun follows two fresh-review class repairs: live admission now requires an identity-bound provider-issued
capability, and provider execution pins and re-verifies the projected primitive before trusting mechanism counters.

**UNKNOWN:** the previous process died without a terminal clean-room receipt, so the exact enriched-minus-clean
delta is UNKNOWN. MLX parity is SKIPPED because Metal was unavailable; NumPy/Torch provider tests are the local
portable boundary.

**VERDICT · fresh-eyes-reviewed(3)-CLEAN: NO-GO.** `verdict_scope=n=1 pair0 at three sealed CE/tau/L7 states
on macOS-CPU advisory; exact frozen-SegNet forward; adaptive projection of eligible 1x1 Conv2d input adjoints;
three timing samples; score_claim=false.` The formulation retains the exact forward and has no admitted arm under
the charged-cycle law. The separate `~95%` profile belongs to SegNet forward-plus-backward plus the INR trunk on a stripped
MLX closure; it is not attributed to this forward alone.

## SFESS reconciliation

| Ours, landed clean-room | Official paper/repository surface | What was added or retained |
|---|---|---|
| Exact conditional-Bernoulli DFT sampler; analytic logit score; M=5 leave-one-out control variate; fixed uniform logits; k=1..5; 64 exact cached calls. | Klas Wijk, Ricardo Vinuesa, and Hossein Azizpour · 2024 · *Revisiting Score Function Estimators for k-Subset Sampling* · arXiv:2407.16058; final ICLR 2025 title *SFESS: Score Function Estimators for k-Subset Sampling*, OpenReview:q87GUkdQBm. The official public repository exposes no LICENSE file at its root. | Added learned logits, bias-corrected Adam, deterministic top-k MAP proposal, exact retention gate, M={2,4,5,8,16,32}, and a near-zero-spread update skip. No repository code was copied. |
| Exact sampler and leave-one-out baseline. | The paper uses the conditional-Poisson/DFT score, a multiple-sample leave-one-out control variate, iterative parameter optimization, and N=32 calibration; its practical Gumbel-top-k sample is explicitly approximate. | Retained exact sampling instead of importing the biased approximation. The 64-call envelope makes M=32 a one-update arm. The operator-folded zero-variance rule skips without dividing by a near-zero standard deviation. |

The DFT normalizer is supported by Manuel Fernández and Stuart Williams · 2010 · *Closed-Form Expression for
the Poisson-Binomial Probability Density Function* · DOI:10.1109/TAES.2010.5461658. Adam is supported by
Diederik P. Kingma and Jimmy Ba · 2015 · *Adam: A Method for Stochastic Optimization* · arXiv:1412.6980.

**MEASURED · fresh-eyes-reviewed(3)-CLEAN:** the in-run clean control, learned-logit k5/M4, and k5/M32 all
returned `S=0.19080429731336374`; enriched-minus-clean delta is exactly `0.0 S`. Exact enumeration is
`S=0.19080359202934188`; the remaining gap is `7.052840218513268e-7 S`, above the registered
`1e-12 S` floor. The k5/M4 arm admitted 2 optimizer proposals and rejected 10 without contaminating retained
Adam state. The final receipt SHA-256 is
`6b0726512fde6aee702f4cb501b819c4a8f20e53e114b9c90b08a68d792b97ff`; terminal recomputation preserved it.

**VERDICT · fresh-eyes-reviewed(3)-CLEAN: NO-GO.** `verdict_scope=64-state cached terminal-polish objective;
fixed-k exact conditional sampling; independently derived learned-logit SFESS; not a byte-for-byte unlicensed OSS
comparison.` It remains usable as a cached discrete optimizer. It does not replace the live frozen forward because
each uncached sample needs an exact objective evaluation.

## JRD / Last-Byte reconciliation

| Ours, clean-room | Official reference | What was missed or not import-admissible |
|---|---|---|
| Exhaustive int8 coefficient-prefix clearing with exact receiver, through-R Seg/Pose, and ZIP-byte gates. | Wuyuan Xie, Zhenming Li, Ye Liu, Jian Jin, Yun Song, and Miaohui Wang · 2026 · *The Last Byte: Learning Just Enough for Machine-Oriented Image Compression* · DOI:10.1609/aaai.v40i19.38635. It trains MVR-Net to predict an encoding-QP map. | The methods are not equivalent. No official source repository/license was located. MVR dataset annotation, QP supervision, and the learned predictor are a separate representation project, not missing prefix-search mechanics. |
| Uniform low-plane clearing plus a locally derived Laplace-motivated dead zone. | Shaohui Li, Han Li, Wenrui Dai, Chenglin Li, Junni Zou, and Hongkai Xiong · 2023 · *Learned Progressive Image Compression With Dead-Zone Quantizers* · DOI:10.1109/TCSVT.2022.3229701. | No official source/license was located, so the exact construction was not imported. The local threshold remains DERIVED clean-room. |
| Full-packet re-encode and measured byte ordering. | Yadong Lu, Yinhao Zhu, Yang Yang, Amir Said, and Taco S. Cohen · 2021 · *Progressive Neural Image Compression with Nested Quantization and Latent Ordering* · arXiv:2102.02913. Related DeepHQ is arXiv:2408.12150. | PLONQ's nested grids, refinement stream, conditional coding, and rate-distortion latent order require a new grammar. No official PLONQ source was located. DeepHQ's relevant extensions are research/non-commercial and patent-restricted; nothing was copied. |

**MEASURED · inherited fixture fresh-eyes-reviewed(3):** on the pair0 local fixture, archive bytes fell from
83,905 to 81,154; `d_seg` fell from `0.023157755533854168` to `0.0218505859375`; `d_pose` fell
from `116.59830629690003` to `92.42743674059255`. That fixture remains GO. The V9/v8 task remains blocked
on `eligible_nonlive_v9_v8_payload_missing_or_unresolved`.

**UNKNOWN · reconciliation fresh-eyes-reviewed(3)-CLEAN:** no implementation passed source, license, and
applicability admission, so no OSS-enriched replay exists and its delta is `UNKNOWN_NOT_MEASURED`.

**VERDICT · fresh-eyes-reviewed(3)-CLEAN: NO-GO for task 449 only.** `verdict_scope=frozen-SegNet
forward-plus-backward throughput replacement; JRD is an archive-rate postprocessor and removes zero teacher calls.`
This does not overturn the pair0 rate-fixture GO.

## Consolidated ranking and closed control laws

1. **No measured route clears task 449.** This is the load-bearing result, scoped to the receipts above.
2. **INSTANT: no admitted arm.** Rank is the smallest retained-energy rank plus oversampling 5 on the smaller
   axis. Admission requires positive renderer-gradient cosine above the fp64 dot-product floor, hot-step
   median-minus-MAD above one, and at least one charged `K={2,4,8}` cycle ratio above one. The largest measured
   hot-step lower bound was `1.0353914790745997`; the largest optimistic charged-cycle ratio was
   `0.5888733451533681`.
3. **YOPO: faithful but validation-dominated NO-GO.** K is the fixed ladder {1,2,4}. The fractional recess starts
   at `1e-2`, halves until strict teacher-CE descent and d_seg non-worsening, and completes at fp32 identity.
4. **SFESS: cached-optimizer-only NO-GO for forward replacement.** k is fixed in 1..5; M is
   {2,4,5,8,16,32}; Adam is constant (`1e-4`, `0.9`, `0.999`, `1e-8`); proposals require exact improvement
   greater than `1e-12 S`; groups with spread at or below `1e-12 S` skip.
5. **JRD: rate-only and outside the throughput mechanism.** Admission is an event-conditioned exact predicate:
   receiver parse-back, strictly fewer ZIP bytes, and componentwise non-worse exact through-R Seg/Pose.

**DERIVED · fresh-eyes-reviewed(3)-CLEAN:** because YOPO still needs costly exact validation, INSTANT and YOPO
retain the exact forward, SFESS adds objective forwards, and JRD removes no scorer call, the next throughput
mechanism must replace the forward amortized over its on-policy training. The registered local rule is on-policy distillation on the witness's
own renders with descent-direction admission, not fixed offline logit matching. If it samples reward groups, the
operator-folded leave-one-out baseline is permitted and near-zero-variance groups must skip. This is a route
classification, not a measured surrogate verdict.

## Triality and task hooks

- YOPO: existing DSL `tac.witness_dsl.scorer_gradient_policy`; equation
  `yopo_first_layer_costate_v1`; task `449_yopo_first_layer_costate_probe_20260712` retains completed/green.
- INSTANT: DSL mode `instant_projected_adjoint` with shared typed `InstantAdmissionEconomics`; equation
  `instant_projected_input_adjoint_v1`; task `449_instant_projected_adjoint_probe_20260712` is completed/green.
- SFESS: existing DSL `tac.witness_dsl.sfess_cached_replay_policy`; equation
  `sfess_fixed_k_cached_replay_ranking_v1` has the learned-logit/M-ladder anchor; task
  `sfess_cached_replay_ugc64_20260712` preserves its existing blocked/green state.
- JRD: DSL leg is N/A because it is an offline receiver/rate oracle, not a trainer/curriculum/actuator lever;
  equation `jrd_exact_coefficient_prefix_selection_v1`; task `v9_jrd_coeff_prefix_probe_20260712` preserves
  blocked/green and its payload blocker.

The shared DAG receives exactly one master FEED row for this pass. The task ledger uses append-only notes and does
not coerce prior blockers or statuses.

## Artifact custody

- YOPO: `experiments/results/yopo_oss_reconciliation_20260713T011432Z/evidence.json`, SHA-256
  `406e77c9323bdf74cf153f7a31c051631a1d1853e9fbf906453e5cd4bd4b5b1a`.
- INSTANT: `experiments/results/instant_oss_reconciliation_20260713T033944Z/measurement_receipt.json`, SHA-256
  `09dbd49c8410c23f2e312670ff517c873e793cde7e64a88fb2df3768586b0443`.
- SFESS: `experiments/results/sfess_oss_reconciliation_20260713T024020Z/measurement_receipt.json`, SHA-256
  `6b0726512fde6aee702f4cb501b819c4a8f20e53e114b9c90b08a68d792b97ff`.
- JRD: `experiments/results/jrd_oss_reconciliation_20260713T011930Z/reconciliation_receipt.json`, SHA-256
  `8cd1f098e84033dcd0aec45a1a5727d9c054809301065f0dcd0a2527fb9c5a1a`.
- SHARE_GE2: `experiments/results/share_ge2_linearity_gate_20260713T013350Z/receipt.json`, SHA-256
  `945302554de5bbd60cefd74dc5e6d116a04d36b0e235d63e7ee9046aa53d2e60`.

## Single DAG FEED row

`FEED-frozen-segnet-oss-master-449 — OSS reconciliation adds the INSTANT adaptive 1x1-conv
input-adjoint tournament and SFESS learned-logit/M-ladder/zero-spread controls; YOPO and JRD import nothing because
their visible official source/license surfaces do not admit copying, and JRD is not the Last-Byte algorithm.
MEASURED: YOPO remains validation-dominated; INSTANT admits no arm after charged validation and retains the exact forward;
SFESS enriched-minus-clean delta is 0.0 S; JRD enriched delta is UNKNOWN. SHARE_GE2 is UNKNOWN because beta_i and
tau are not mapped to SegNet Jacobians, so exact-teacher descent and validation economics decide. VERDICT: none
clears #449 in the registered scopes; route to an amortized on-policy nonlinear forward-replacing surrogate.
score_claim=false; pointer 0.1880443979880752 unmoved; final review provenance is carried by verifier-authored
master receipts.`
