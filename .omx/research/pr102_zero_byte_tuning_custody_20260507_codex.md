# PR102 Zero-Byte Tuning Custody

timestamp_utc: 2026-05-07T15:47:30Z
score_claim: false
dispatch_attempted: false
ready_for_exact_eval_dispatch: false

## Finding

PR102 is a zero-byte runtime-constant change over PR100, but the existing local
PR102 generic intake captured the wrong release asset:

- wrong local archive:
  `experiments/results/public_pr_intake_full/public_pr102_intake_20260505_auto/archive.zip`
- wrong archive bytes: `276481`
- wrong archive SHA-256:
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
- wrong member: `p`

The correct HNeRV archive is the release asset referenced by the PR102 source
and is byte-identical to PR100:

- correct archive:
  `experiments/results/pr102_zero_byte_tuning_custody_20260507_codex/ethan_v2_hnerv_lc_scale095_archive.zip`
- correct archive bytes: `178981`
- correct archive SHA-256:
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- correct member: `0.bin`
- byte-identical to:
  `experiments/results/public_pr100_intake_20260504_codex/archive.zip`

## Runtime Changes

The PR102 source changes runtime behavior only:

- `sidecar.py`: `DELTA_SCALE = 0.0095`
- `inflate.py`: `up[:, 0, 0].add_(1.0)` after bicubic upsampling

Archive byte delta versus PR100 is exactly `0`.

## Artifact

- audit tool: `tools/audit_pr102_zero_byte_tuning_custody.py`
- test: `src/tac/tests/test_audit_pr102_zero_byte_tuning_custody.py`
- manifest:
  `experiments/results/pr102_zero_byte_tuning_custody_20260507_codex/custody_manifest.json`

The manifest marks:

- `ready_for_source_schema_review=true`
- `ready_for_exact_eval_dispatch=false`
- `existing_pr102_intake_archive_wrong=true`

## Blockers

- `existing_pr102_intake_archive_is_wrong_release_asset`
- `pr102_exact_cuda_replay_missing`
- `pr102_port_to_current_stack_missing`
- `no_op_control_missing`

This promotes PR102 from "stale/wrong local archive" to a clean source-custody
target. It does not rank PR102 locally or claim score until exact CUDA replay or
a current-stack port plus no-op control exists.
