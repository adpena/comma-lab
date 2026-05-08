# Sub-0.17 CPU Frontier Plan

score_claim: `false`
ready_for_exact_eval_dispatch: `false`
target_score: `0.17`

## Byte Budget

- anchor archive bytes: `185578`
- max archive bytes if components hold: `127035`
- required archive savings: `58543`
- target weight payload bytes: `111078`

## Recommended Candidate

- id: `svd_stem_blocks012_balanced`
- estimated archive bytes: `111347`
- projected score if components hold: `0.159553699633`
- risk: `medium_high`

## Safe Stacking Order

1. `svd_low_rank_stem_and_early_conv2d` - remove large decoder weight mass before quantization noise is introduced
2. `continuous_k_allocation` - choose per-stream K under a global distortion budget after SVD changes tensor spectra
3. `analytical_lossy_coarsening` - coarsen factor and residual streams only after rank selection
4. `entropy_pack_and_noop_guards` - pack final charged streams and reject no-op or metadata-only wins

## Dispatch Blockers

- `no_archive_built_by_this_tool`
- `cpu_projection_not_score_evidence`
- `factorized_hnerv_runtime_not_implemented`
- `exact_cuda_auth_eval_required_before_score_claim`
- `remote_dispatch_forbidden_for_this_task`
