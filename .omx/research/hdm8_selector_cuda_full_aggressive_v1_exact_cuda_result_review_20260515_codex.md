# HDM8 Selector CUDA Full Aggressive v1 Exact-CUDA Result Review - 2026-05-15

## Scope

Review the exact `[contest-CUDA]` result for the byte-closed HDM8/PR106
postdecode selector packet built from the full 600-pair Modal T4 CUDA-prefix
sweep.

This review answers whether the film-grain/postfilter selector and dynamic
water-fill candidate should be promoted, retried, or classified as a measured
configuration negative.

## Candidate

- archive: `experiments/results/hdm8_selector_cuda_full_aggressive_v1_clean_20260515_codex/archive.zip`
- archive_sha256: `34dc94644f5619ea7e6254079e3e4d3bbf0952f8a0ad287f675f7a249f359071`
- archive_bytes: `187226`
- source archive: `experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip`
- source exact-CUDA reference: `experiments/results/modal_auth_eval/hnerv_hdm8_fixed_lengths_modal_t4_retry2_20260514T095000Z/contest_auth_eval.json`
- source archive_sha256: `8a30730e863a2f846d7ca3a707b3191ad64312f5270976dc5f9322ba4228e8c2`
- source archive_bytes: `186395`
- selector bytes charged in archive: `829`
- archive byte delta vs source: `831`
- runtime_tree_sha256: `43cd451d441cc134738435957ae14be43f3a17907eab10991d7da4638c050572`
- runtime_content_tree_sha256: `472e7241ca53641a70b9c28772fbcfe99548534c210ebaaa684194e91f745dd2`
- repo commit: `d8b841d31f398f455796586100df3c5b37c9aa2a`
- dirty at packet build: `false`

## Prefix Evidence Before Exact Eval

Full 600-pair Modal T4 CUDA-prefix sweep:

- sweep: `experiments/results/modal_hdm8_postfilter_sweep/hdm8_cuda_full_aggressive_v1_fix1_20260515T023053Z/hdm8_postfilter_sweep.json`
- call_id: `fc-01KRMQJPC9RTEZCMP3Z5B95MY6`
- axis: `modal-t4-cuda-proxy-prefix`
- n_pairs: `600`
- baseline_score_proxy: `0.22779605460207308`
- selector_score_proxy_charged: `0.22202386779659933`
- charged proxy delta: `-0.005772186805473756`
- baseline_avg_posenet_dist: `0.0001640114178492998`
- selector_avg_posenet_dist: `0.00011677807693104114`
- baseline_avg_segnet_dist: `0.00063184951878308`
- selector_avg_segnet_dist: `0.0006318495198502206`
- gate: `passed_cuda_prefix_component_check`

The prefix evidence was positive and sufficient to dispatch exact CUDA, but it
was explicitly `score_claim=false` and `promotion_eligible=false`.

## Exact CUDA Result

Modal T4 exact auth eval:

- call_id: `fc-01KRMR0WTB9G69ZSW1SM37H5NB`
- output_dir: `experiments/results/modal_auth_eval/hdm8_selector_cuda_full_aggressive_v1_clean_20260515T023845Z`
- eval JSON: `experiments/results/modal_auth_eval/hdm8_selector_cuda_full_aggressive_v1_clean_20260515T023845Z/contest_auth_eval.json`
- inflated output aggregate sha256: `0da4f6c994375f7062975a1170c9e1d5e553fac70060f75ee6a0d0698a12b53d`
- samples: `600`
- avg_segnet_dist: `0.0006426`
- avg_posenet_dist: `0.00004241`
- canonical score: `0.2095197967107254`
- reported rounded score: `0.21`
- evidence_grade: `contest-CUDA`
- score_axis: `contest_cuda`

Formula check:

```text
100 * 0.0006426
+ sqrt(10 * 0.00004241)
+ 25 * 187226 / 37545489
= 0.2095197967107254
```

Reference exact-CUDA result:

- eval JSON: `experiments/results/modal_auth_eval/hnerv_hdm8_fixed_lengths_modal_t4_retry2_20260514T095000Z/contest_auth_eval.json`
- avg_segnet_dist: `0.0006426`
- avg_posenet_dist: `0.00003236`
- archive_bytes: `186395`
- canonical score: `0.20636166502462222`

Delta vs reference:

- score delta: `+0.0031581316861031827` (worse)
- pose-term delta: `+0.002604802896058654`
- seg-term delta: `0.0`
- rate-term delta: `+0.0005533287900445216`
- archive byte delta: `+831`

## Classification

`measured_config_regression`

The byte-closed selector is not promotion-eligible. The exact CUDA result is
worse than the HDM8 reference because the selector adds 831 bytes and increases
PoseNet distortion, while SegNet is unchanged at reported precision.

This is not evidence that film-grain/postdecode selectors are exhausted. It is
evidence that the current full-sweep selector objective is not exact-CUDA
robust:

- The prefix proxy predicted a PoseNet improvement.
- Exact CUDA measured a PoseNet regression relative to the reference.
- SegNet did not move materially, so the useful target is PoseNet-safe
  postdecode tuning, not unconstrained selector expansion.
- The packed selector overhead is small enough to keep the lane alive only if
  exact CUDA component movement is positive.

## Next Engineering Action

Do not promote this packet.

Continue the family only under stricter exact-CUDA controls:

1. Build a paired selector objective that ranks pair/mode choices against exact
   CUDA deltas from held-out exact-eval probes instead of prefix-only local
   linearization.
2. Add a selector sparsity/byte budget term so pair choices must beat the
   measured 831-byte overhead, not just proxy component score.
3. Prioritize modes that are PoseNet-neutral or PoseNet-positive on exact CUDA;
   keep SegNet-flat modes only when the rate term is already paid by another
   mechanism.
4. Treat CPU/MPS-positive PR101 selector results as separate-axis evidence; do
   not transfer them to CUDA without paired exact CUDA or full-frame/runtime
   parity plus exact scorer confirmation.

## Status

- exact CUDA recovered: yes
- terminal lane claim recorded: yes
- score claim valid for this exact CUDA artifact: yes
- promotion_eligible: no
- rank_or_kill_eligible for the family: no
- score_claim for future selector variants: false until byte-closed exact CUDA
  passes against the matched baseline
