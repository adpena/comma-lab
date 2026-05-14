# C6 decoder_blob:34858 Exact CUDA Result - 2026-05-14

## Result

- lane_id: `lane_c6_ibps1_decoder34858_byte_patch_exact_cuda_20260514`
- instance_job_id: `c6_ibps1_decoder34858_patch_t4_20260514T182251Z`
- Modal call_id: `fc-01KRKVR2VK30AVTW35J5SCAW4Z`
- score_axis: `contest_cuda`
- evidence_grade: `contest-CUDA`
- archive_sha256: `2d6416874c6563d6f2ebf9b502c98a45b5b4dfee88f42124dbbaa1910579bb3a`
- archive_size_bytes: `224481`
- avg_segnet_dist: `0.50482631`
- avg_posenet_dist: `0.43463999`
- score_recomputed_from_components: `52.7169058085588`
- n_samples: `600`
- hardware: Linux x86_64, Tesla T4, scorer `cuda`, inflate device `auto`

## Artifact paths

- result: `experiments/results/modal_auth_eval/c6_ibps1_decoder34858_patch_t4_20260514T182251Z/modal_cuda_auth_eval_result.json`
- auth eval: `experiments/results/modal_auth_eval/c6_ibps1_decoder34858_patch_t4_20260514T182251Z/contest_auth_eval.json`
- validation: `experiments/results/modal_auth_eval/c6_ibps1_decoder34858_patch_t4_20260514T182251Z/modal_cuda_auth_eval_validation.json`
- compliance: `experiments/results/modal_auth_eval/c6_ibps1_decoder34858_patch_t4_20260514T182251Z/pre_submission_compliance_with_auth_eval.json`
- inflated output manifest: `experiments/results/modal_auth_eval/c6_ibps1_decoder34858_patch_t4_20260514T182251Z/inflated_outputs_manifest.json`
- packet: `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_34858/submission_dir/archive.zip`

## Compliance

`scripts/pre_submission_compliance_check.py` passed with:

- `--require-auth-eval`
- `--require-t4-equivalent`
- `--require-submission-runtime-match`
- `--expected-runtime-tree-sha256 03c0788a1fb2cc4842803135820b350cf5db62f598403954996faef972c56f98`
- matching lane/job claim evidence in `.omx/state/active_lane_dispatch_claims.md`

## Classification

This is a valid byte-closed exact-CUDA artifact, but it is not a frontier or
promotion candidate. The absolute score is dominated by the immature C6 5ep
substrate quality. Treat it as:

- exact evidence that the C6 patch packet inflates and scores under the contest
  CUDA path;
- a negative for current-score competitiveness;
- not sufficient to prove the single-byte patch improves the exact-CUDA source
  archive, because the unpatched 5ep source archive was not evaluated under the
  same exact-CUDA runtime.

The useful next move is not more single-byte patches on this 5ep substrate.
Either exact-evaluate the unpatched source only if a clean delta is required, or
move C6 effort to converged training / latent-consumption repair before more
paid exact eval.
