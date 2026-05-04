# PR85 Non-Mask Self-Compression Audit

- planning_only: true
- score_claim: false
- dispatch_performed: false
- archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- archive_bytes: 236328
- x_member_bytes: 236228
- zip_overhead_bytes: 100

## Direct Lossless Candidates

- none: all non-mask decoded Brotli recodes and outer-container baselines were non-improving or blocked.

## Architecture Transfer Candidates

- `pr90_qfq4_style_pr85_model_serializer_probe`: -689 bytes (formula-only rate delta -0.000458776819), runtime risk high; PR90 model body is smaller than PR85 model segment, but not byte-identical or runtime-compatible.
- `pr90_qrgb_control_stack_recode_probe`: -10955 bytes (formula-only rate delta -0.007294484831), runtime risk high; PR90 compact control body is smaller than the PR85 non-mask control stack, but semantics are different.

## PR91 Identity

- available: True
- nonmask_all_identity: True
- conclusion: PR91 changes the mask segment only; any true PR85 non-mask byte reduction should stack byte-for-byte with PR91.
