# Z3 Full Modal Harvest - 2026-05-15

## Result

- lane_id: `lane_z3_balle_hyperprior_bolton_recover_20260514`
- job: `substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T114115Z__full__1000ep`
- Modal call_id: `fc-01KRNQ2B8ZH6ASSAHCXMH0AMZX`
- terminal claim: `completed_modal_training_recovered_no_score_claim`
- artifact root: `experiments/results/lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T114115Z__full__1000ep_modal/`
- cost anchor: `$0.1733562924935139` estimated, `T4`, `1057.7672084350002` seconds

## Advisory Auth Eval

The recovered wrapper emitted a diagnostic CPU-axis auth-eval artifact. This is
not promotion-ready and not rank/kill eligible until the byte-closed archive is
replayed through the exact contest CUDA path.

- advisory canonical score: `0.19869364567798808`
- displayed rounded score: `0.2`
- score_axis: `diagnostic_cpu`
- evidence_grade: `B`
- evidence_semantics: `diagnostic_auth_eval_non_promotable`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- exact_cuda_eval_complete: `false`
- diagnostic_blockers: `modal_training_wrapper_auth_eval_advisory_only`
- n_samples: `600`
- archive_size_bytes: `179130`
- archive_sha256: `b6c4a6f1f1f4bb29695e8ee095ca3862690b2c4833fba31579406179aaf35a4b`
- avg_segnet_dist: `0.0006141`
- avg_posenet_dist: `0.00003243`
- score_seg_contribution: `0.06141`
- score_pose_contribution: `0.018008331405213532`
- score_rate_contribution: `0.11927531427277456`

## Classification

This is a useful Z3 training/throughput and byte-closed archive anchor, but not
a sub-0.192 candidate and not a promotion result. The correct next score-facing
action is exact CUDA replay of the recovered archive only if Z3 remains on the
frontier queue after higher-EV paired CPU/CUDA candidates are processed.

## Verification

- recovered artifacts with `experiments/modal_recover_lane.py --call-id fc-01KRNQ2B8ZH6ASSAHCXMH0AMZX`
- auth artifact:
  `experiments/results/lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T114115Z__full__1000ep_modal/lane_substrate_z3_balle_hyperprior_bolton_results/output/contest_auth_eval_cuda.json`
- archive:
  `experiments/results/lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T114115Z__full__1000ep_modal/lane_substrate_z3_balle_hyperprior_bolton_results/output/archive.zip`

