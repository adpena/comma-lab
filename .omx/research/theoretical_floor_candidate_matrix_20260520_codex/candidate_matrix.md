# Source-Backed Theoretical-Floor Candidate Matrix

Authority: design/runtime routing only; score_claim=false.

## Baseline Correction

- weighted SegNet term: `0.056029`
- weighted PoseNet term: `0.017155`
- weighted rate term: `0.118867`
- distortion-only zero-archive score: `0.073184`
- rate-only perfect-distortion score: `0.118867`

## Candidate Contracts

| candidate | runtime payload | teachers | first gate | blockers |
|---|---|---|---|---|
| tf_siren_first_anchor | 0.bin, inflate.py, inflate.sh | score_aware_loss | `.venv/bin/python experiments/train_substrate_siren.py --video-path upstream/videos/0.mkv --output-dir experiments/results/siren_smoke_<utc> --epochs 3 --device cpu --smoke --skip-archive-build --skip-auth-eval` | operator_authorization_required_for_gpu_spend, active_lane_dispatch_claim_required, exact_cuda_auth_eval_missing |
| tf_telescope_lfv1_pose_foveation | lapose_foveation_tuples.lfv1, foveation_params.bin, runtime_consumer.py, inflate.sh | telescope_hyperbolic_foveation, la_pose, sea_raft, visual_primitives | `.venv/bin/python tools/build_lapose_foveation_payload_archive.py --out-dir experiments/results/theoretical_floor_lfv1_<utc>/archive_candidate --lfv1-payload experiments/results/theoretical_floor_lfv1_<utc>/lapose_foveation_tuples.lfv1 --source-readiness-json experiments/results/theoretical_floor_lfv1_<utc>/lfv1_payload_readiness.json` | runtime_loader_parity_not_passed, scorer_visible_output_parity_not_proven, runtime_output_parity_not_proven, exact_cuda_auth_eval_missing |
| tf_vqvae_full_renderer | 0.bin, inflate.py, inflate.sh | score_aware_loss, codebook_perplexity_gate | `.venv/bin/python -m pytest src/tac/tests/test_train_vqvae_as_renderer.py src/tac/tests/test_vqvae_mask_codec.py -q` | byte_closed_export_smoke_missing, exact_cuda_auth_eval_missing |
| tf_c3_coolchic_sparse_residual | archive.zip, residual_bytes | cool_chic, c3, hinton_distilled_scorer | `Run l2_encoded materializer with explicit --decoded-raw, --gt-raw, --byte-budget, --encoding sparse, and no score claim.` | decoded_raw_required, gt_raw_required, explicit_byte_budget_required, exact_cuda_auth_eval_missing |

## Source References

- `telescope_hyperbolic_foveation`: Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection (https://princeton-computational-imaging.github.io/Telescope/); inflate-time dependency allowed: `false`
- `la_pose`: LA-Pose: Latent Action Pretraining Meets Pose Estimation (https://arxiv.org/abs/2604.27448); inflate-time dependency allowed: `false`
- `sea_raft`: SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow (https://arxiv.org/abs/2405.14793); inflate-time dependency allowed: `false`
- `visual_primitives`: Thinking with Visual Primitives (https://www.k-a.in/Thinking_with_Visual_Primitives.pdf); inflate-time dependency allowed: `false`
- `siren`: Implicit neural representation / SIREN family (https://arxiv.org/abs/2006.09661); inflate-time dependency allowed: `true`
- `vq_vae`: Neural Discrete Representation Learning / VQ-VAE family (https://arxiv.org/abs/1711.00937); inflate-time dependency allowed: `true`
