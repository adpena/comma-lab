# L5 v2 TT5L Lightning non-dry-run gate

Generated: 2026-05-18T06:15:28Z

This gate is spend-readiness only. It does not dispatch provider work and does not claim score movement. It fails closed unless the doctor output is OK, every per-cell source manifest has a real remote-verified staging receipt, every cell has an active Lightning lane claim, and all non-dry-run submit templates are free of placeholders.

## Status

- Source bundle: `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json`
- Doctor plan: `.omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.json`
- Doctor output: `.omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json`
- Claims ledger: `.omx/state/active_lane_dispatch_claims.md`
- ready_for_non_dry_run_submit: `False`
- ready_for_provider_dispatch: `False`
- score_claim: `false`
- promotion_eligible: `false`
- dispatch_attempted: `false`
- Ready cells: `0`/`10`
- Blocker count: `166`

## Top-Level Blockers

- `doctor_output_status_not_ok`
- `doctor_output_failed_checks_not_empty`
- `doctor_check_not_ok:ssh_auth`
- `doctor_check_not_ok:remote_supply_chain`
- `doctor_check_not_ok:machine_inventory`
- `doctor_machine_inventory_empty`
- `zero:contest_cpu:source_manifest:json_file_missing`
- `zero:contest_cpu:source_manifest_missing`
- `zero:contest_cpu:source_manifest_invalid_or_empty`
- `zero:contest_cpu:source_manifest_artifact_paths_missing`
- `zero:contest_cpu:source_manifest_git_head_mismatch_bundle`
- `zero:contest_cpu:source_bundle_current_head_mismatch`
- `zero:contest_cpu:source_manifest_git_head_mismatch_current`
- `zero:contest_cpu:source_manifest_archive_file_missing`
- `zero:contest_cpu:stage_receipt_missing`
- `zero:contest_cpu:stage_receipt_invalid_or_empty`
- `zero:contest_cpu:non_dry_run_command_placeholders_present`
- `zero:contest_cpu:non_dry_run_command_arg_placeholder:--studio`
- `zero:contest_cpu:non_dry_run_command_arg_placeholder:--teamspace`
- `zero:contest_cpu:non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target`
- `zero:contest_cpu:non_dry_run_command_identity_mode_not_exactly_one`
- `zero:contest_cpu:active_lightning_claim_missing`
- `zero:contest_cuda:source_manifest:json_file_missing`
- `zero:contest_cuda:source_manifest_missing`
- `zero:contest_cuda:source_manifest_invalid_or_empty`
- `zero:contest_cuda:source_manifest_artifact_paths_missing`
- `zero:contest_cuda:source_manifest_git_head_mismatch_bundle`
- `zero:contest_cuda:source_bundle_current_head_mismatch`
- `zero:contest_cuda:source_manifest_git_head_mismatch_current`
- `zero:contest_cuda:source_manifest_archive_file_missing`
- `zero:contest_cuda:stage_receipt_missing`
- `zero:contest_cuda:stage_receipt_invalid_or_empty`
- `zero:contest_cuda:non_dry_run_command_placeholders_present`
- `zero:contest_cuda:non_dry_run_command_arg_placeholder:--studio`
- `zero:contest_cuda:non_dry_run_command_arg_placeholder:--teamspace`
- `zero:contest_cuda:non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target`
- `zero:contest_cuda:non_dry_run_command_identity_mode_not_exactly_one`
- `zero:contest_cuda:active_lightning_claim_missing`
- `random_lsb:contest_cpu:source_manifest:json_file_missing`
- `random_lsb:contest_cpu:source_manifest_missing`
- `random_lsb:contest_cpu:source_manifest_invalid_or_empty`
- `random_lsb:contest_cpu:source_manifest_artifact_paths_missing`
- `random_lsb:contest_cpu:source_manifest_git_head_mismatch_bundle`
- `random_lsb:contest_cpu:source_bundle_current_head_mismatch`
- `random_lsb:contest_cpu:source_manifest_git_head_mismatch_current`
- `random_lsb:contest_cpu:source_manifest_archive_file_missing`
- `random_lsb:contest_cpu:stage_receipt_missing`
- `random_lsb:contest_cpu:stage_receipt_invalid_or_empty`
- `random_lsb:contest_cpu:non_dry_run_command_placeholders_present`
- `random_lsb:contest_cpu:non_dry_run_command_arg_placeholder:--studio`
- `random_lsb:contest_cpu:non_dry_run_command_arg_placeholder:--teamspace`
- `random_lsb:contest_cpu:non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target`
- `random_lsb:contest_cpu:non_dry_run_command_identity_mode_not_exactly_one`
- `random_lsb:contest_cpu:active_lightning_claim_missing`
- `random_lsb:contest_cuda:source_manifest:json_file_missing`
- `random_lsb:contest_cuda:source_manifest_missing`
- `random_lsb:contest_cuda:source_manifest_invalid_or_empty`
- `random_lsb:contest_cuda:source_manifest_artifact_paths_missing`
- `random_lsb:contest_cuda:source_manifest_git_head_mismatch_bundle`
- `random_lsb:contest_cuda:source_bundle_current_head_mismatch`
- `random_lsb:contest_cuda:source_manifest_git_head_mismatch_current`
- `random_lsb:contest_cuda:source_manifest_archive_file_missing`
- `random_lsb:contest_cuda:stage_receipt_missing`
- `random_lsb:contest_cuda:stage_receipt_invalid_or_empty`
- `random_lsb:contest_cuda:non_dry_run_command_placeholders_present`
- `random_lsb:contest_cuda:non_dry_run_command_arg_placeholder:--studio`
- `random_lsb:contest_cuda:non_dry_run_command_arg_placeholder:--teamspace`
- `random_lsb:contest_cuda:non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target`
- `random_lsb:contest_cuda:non_dry_run_command_identity_mode_not_exactly_one`
- `random_lsb:contest_cuda:active_lightning_claim_missing`
- `shuffled:contest_cpu:source_manifest:json_file_missing`
- `shuffled:contest_cpu:source_manifest_missing`
- `shuffled:contest_cpu:source_manifest_invalid_or_empty`
- `shuffled:contest_cpu:source_manifest_artifact_paths_missing`
- `shuffled:contest_cpu:source_manifest_git_head_mismatch_bundle`
- `shuffled:contest_cpu:source_bundle_current_head_mismatch`
- `shuffled:contest_cpu:source_manifest_git_head_mismatch_current`
- `shuffled:contest_cpu:source_manifest_archive_file_missing`
- `shuffled:contest_cpu:stage_receipt_missing`
- `shuffled:contest_cpu:stage_receipt_invalid_or_empty`
- `shuffled:contest_cpu:non_dry_run_command_placeholders_present`
- `shuffled:contest_cpu:non_dry_run_command_arg_placeholder:--studio`
- `shuffled:contest_cpu:non_dry_run_command_arg_placeholder:--teamspace`
- `shuffled:contest_cpu:non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target`
- `shuffled:contest_cpu:non_dry_run_command_identity_mode_not_exactly_one`
- `shuffled:contest_cpu:active_lightning_claim_missing`
- `shuffled:contest_cuda:source_manifest:json_file_missing`
- `shuffled:contest_cuda:source_manifest_missing`
- `shuffled:contest_cuda:source_manifest_invalid_or_empty`
- `shuffled:contest_cuda:source_manifest_artifact_paths_missing`
- `shuffled:contest_cuda:source_manifest_git_head_mismatch_bundle`
- `shuffled:contest_cuda:source_bundle_current_head_mismatch`
- `shuffled:contest_cuda:source_manifest_git_head_mismatch_current`
- `shuffled:contest_cuda:source_manifest_archive_file_missing`
- `shuffled:contest_cuda:stage_receipt_missing`
- `shuffled:contest_cuda:stage_receipt_invalid_or_empty`
- `shuffled:contest_cuda:non_dry_run_command_placeholders_present`
- `shuffled:contest_cuda:non_dry_run_command_arg_placeholder:--studio`
- `shuffled:contest_cuda:non_dry_run_command_arg_placeholder:--teamspace`
- `shuffled:contest_cuda:non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target`
- `shuffled:contest_cuda:non_dry_run_command_identity_mode_not_exactly_one`
- `shuffled:contest_cuda:active_lightning_claim_missing`
- `trained:contest_cpu:source_manifest:json_file_missing`
- `trained:contest_cpu:source_manifest_missing`
- `trained:contest_cpu:source_manifest_invalid_or_empty`
- `trained:contest_cpu:source_manifest_artifact_paths_missing`
- `trained:contest_cpu:source_manifest_git_head_mismatch_bundle`
- `trained:contest_cpu:source_bundle_current_head_mismatch`
- `trained:contest_cpu:source_manifest_git_head_mismatch_current`
- `trained:contest_cpu:source_manifest_archive_file_missing`
- `trained:contest_cpu:stage_receipt_missing`
- `trained:contest_cpu:stage_receipt_invalid_or_empty`
- `trained:contest_cpu:non_dry_run_command_placeholders_present`
- `trained:contest_cpu:non_dry_run_command_arg_placeholder:--studio`
- `trained:contest_cpu:non_dry_run_command_arg_placeholder:--teamspace`
- `trained:contest_cpu:non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target`
- `trained:contest_cpu:non_dry_run_command_identity_mode_not_exactly_one`
- `trained:contest_cpu:active_lightning_claim_missing`
- `trained:contest_cuda:source_manifest:json_file_missing`
- `trained:contest_cuda:source_manifest_missing`
- `trained:contest_cuda:source_manifest_invalid_or_empty`
- `trained:contest_cuda:source_manifest_artifact_paths_missing`
- `trained:contest_cuda:source_manifest_git_head_mismatch_bundle`
- `trained:contest_cuda:source_bundle_current_head_mismatch`
- `trained:contest_cuda:source_manifest_git_head_mismatch_current`
- `trained:contest_cuda:source_manifest_archive_file_missing`
- `trained:contest_cuda:stage_receipt_missing`
- `trained:contest_cuda:stage_receipt_invalid_or_empty`
- `trained:contest_cuda:non_dry_run_command_placeholders_present`
- `trained:contest_cuda:non_dry_run_command_arg_placeholder:--studio`
- `trained:contest_cuda:non_dry_run_command_arg_placeholder:--teamspace`
- `trained:contest_cuda:non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target`
- `trained:contest_cuda:non_dry_run_command_identity_mode_not_exactly_one`
- `trained:contest_cuda:active_lightning_claim_missing`
- `ablated:contest_cpu:source_manifest:json_file_missing`
- `ablated:contest_cpu:source_manifest_missing`
- `ablated:contest_cpu:source_manifest_invalid_or_empty`
- `ablated:contest_cpu:source_manifest_artifact_paths_missing`
- `ablated:contest_cpu:source_manifest_git_head_mismatch_bundle`
- `ablated:contest_cpu:source_bundle_current_head_mismatch`
- `ablated:contest_cpu:source_manifest_git_head_mismatch_current`
- `ablated:contest_cpu:source_manifest_archive_file_missing`
- `ablated:contest_cpu:stage_receipt_missing`
- `ablated:contest_cpu:stage_receipt_invalid_or_empty`
- `ablated:contest_cpu:non_dry_run_command_placeholders_present`
- `ablated:contest_cpu:non_dry_run_command_arg_placeholder:--studio`
- `ablated:contest_cpu:non_dry_run_command_arg_placeholder:--teamspace`
- `ablated:contest_cpu:non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target`
- `ablated:contest_cpu:non_dry_run_command_identity_mode_not_exactly_one`
- `ablated:contest_cpu:active_lightning_claim_missing`
- `ablated:contest_cuda:source_manifest:json_file_missing`
- `ablated:contest_cuda:source_manifest_missing`
- `ablated:contest_cuda:source_manifest_invalid_or_empty`
- `ablated:contest_cuda:source_manifest_artifact_paths_missing`
- `ablated:contest_cuda:source_manifest_git_head_mismatch_bundle`
- `ablated:contest_cuda:source_bundle_current_head_mismatch`
- `ablated:contest_cuda:source_manifest_git_head_mismatch_current`
- `ablated:contest_cuda:source_manifest_archive_file_missing`
- `ablated:contest_cuda:stage_receipt_missing`
- `ablated:contest_cuda:stage_receipt_invalid_or_empty`
- `ablated:contest_cuda:non_dry_run_command_placeholders_present`
- `ablated:contest_cuda:non_dry_run_command_arg_placeholder:--studio`
- `ablated:contest_cuda:non_dry_run_command_arg_placeholder:--teamspace`
- `ablated:contest_cuda:non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target`
- `ablated:contest_cuda:non_dry_run_command_identity_mode_not_exactly_one`
- `ablated:contest_cuda:active_lightning_claim_missing`

## Cells

| variant | axis | lane_id | job_name | ready | blockers |
| --- | --- | --- | --- | --- | --- |
| `zero` | `[contest-CPU]` | `lane_l5_v2_tt5l_sideinfo_effect_curve_zero_contest_cpu` | `l5-v2-tt5l-sideinfo-zero-cpu-20260517` | `False` | `['source_manifest:json_file_missing', 'source_manifest_missing', 'source_manifest_invalid_or_empty', 'source_manifest_artifact_paths_missing', 'source_manifest_git_head_mismatch_bundle', 'source_bundle_current_head_mismatch', 'source_manifest_git_head_mismatch_current', 'source_manifest_archive_file_missing', 'stage_receipt_missing', 'stage_receipt_invalid_or_empty', 'non_dry_run_command_placeholders_present', 'non_dry_run_command_arg_placeholder:--studio', 'non_dry_run_command_arg_placeholder:--teamspace', 'non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target', 'non_dry_run_command_identity_mode_not_exactly_one', 'active_lightning_claim_missing']` |
| `zero` | `[contest-CUDA]` | `lane_l5_v2_tt5l_sideinfo_effect_curve_zero_contest_cuda` | `l5-v2-tt5l-sideinfo-zero-cuda-20260517` | `False` | `['source_manifest:json_file_missing', 'source_manifest_missing', 'source_manifest_invalid_or_empty', 'source_manifest_artifact_paths_missing', 'source_manifest_git_head_mismatch_bundle', 'source_bundle_current_head_mismatch', 'source_manifest_git_head_mismatch_current', 'source_manifest_archive_file_missing', 'stage_receipt_missing', 'stage_receipt_invalid_or_empty', 'non_dry_run_command_placeholders_present', 'non_dry_run_command_arg_placeholder:--studio', 'non_dry_run_command_arg_placeholder:--teamspace', 'non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target', 'non_dry_run_command_identity_mode_not_exactly_one', 'active_lightning_claim_missing']` |
| `random_lsb` | `[contest-CPU]` | `lane_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_contest_cpu` | `l5-v2-tt5l-sideinfo-random-lsb-cpu-20260517` | `False` | `['source_manifest:json_file_missing', 'source_manifest_missing', 'source_manifest_invalid_or_empty', 'source_manifest_artifact_paths_missing', 'source_manifest_git_head_mismatch_bundle', 'source_bundle_current_head_mismatch', 'source_manifest_git_head_mismatch_current', 'source_manifest_archive_file_missing', 'stage_receipt_missing', 'stage_receipt_invalid_or_empty', 'non_dry_run_command_placeholders_present', 'non_dry_run_command_arg_placeholder:--studio', 'non_dry_run_command_arg_placeholder:--teamspace', 'non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target', 'non_dry_run_command_identity_mode_not_exactly_one', 'active_lightning_claim_missing']` |
| `random_lsb` | `[contest-CUDA]` | `lane_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_contest_cuda` | `l5-v2-tt5l-sideinfo-random-lsb-cuda-20260517` | `False` | `['source_manifest:json_file_missing', 'source_manifest_missing', 'source_manifest_invalid_or_empty', 'source_manifest_artifact_paths_missing', 'source_manifest_git_head_mismatch_bundle', 'source_bundle_current_head_mismatch', 'source_manifest_git_head_mismatch_current', 'source_manifest_archive_file_missing', 'stage_receipt_missing', 'stage_receipt_invalid_or_empty', 'non_dry_run_command_placeholders_present', 'non_dry_run_command_arg_placeholder:--studio', 'non_dry_run_command_arg_placeholder:--teamspace', 'non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target', 'non_dry_run_command_identity_mode_not_exactly_one', 'active_lightning_claim_missing']` |
| `shuffled` | `[contest-CPU]` | `lane_l5_v2_tt5l_sideinfo_effect_curve_shuffled_contest_cpu` | `l5-v2-tt5l-sideinfo-shuffled-cpu-20260517` | `False` | `['source_manifest:json_file_missing', 'source_manifest_missing', 'source_manifest_invalid_or_empty', 'source_manifest_artifact_paths_missing', 'source_manifest_git_head_mismatch_bundle', 'source_bundle_current_head_mismatch', 'source_manifest_git_head_mismatch_current', 'source_manifest_archive_file_missing', 'stage_receipt_missing', 'stage_receipt_invalid_or_empty', 'non_dry_run_command_placeholders_present', 'non_dry_run_command_arg_placeholder:--studio', 'non_dry_run_command_arg_placeholder:--teamspace', 'non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target', 'non_dry_run_command_identity_mode_not_exactly_one', 'active_lightning_claim_missing']` |
| `shuffled` | `[contest-CUDA]` | `lane_l5_v2_tt5l_sideinfo_effect_curve_shuffled_contest_cuda` | `l5-v2-tt5l-sideinfo-shuffled-cuda-20260517` | `False` | `['source_manifest:json_file_missing', 'source_manifest_missing', 'source_manifest_invalid_or_empty', 'source_manifest_artifact_paths_missing', 'source_manifest_git_head_mismatch_bundle', 'source_bundle_current_head_mismatch', 'source_manifest_git_head_mismatch_current', 'source_manifest_archive_file_missing', 'stage_receipt_missing', 'stage_receipt_invalid_or_empty', 'non_dry_run_command_placeholders_present', 'non_dry_run_command_arg_placeholder:--studio', 'non_dry_run_command_arg_placeholder:--teamspace', 'non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target', 'non_dry_run_command_identity_mode_not_exactly_one', 'active_lightning_claim_missing']` |
| `trained` | `[contest-CPU]` | `lane_l5_v2_tt5l_sideinfo_effect_curve_trained_contest_cpu` | `l5-v2-tt5l-sideinfo-trained-cpu-20260517` | `False` | `['source_manifest:json_file_missing', 'source_manifest_missing', 'source_manifest_invalid_or_empty', 'source_manifest_artifact_paths_missing', 'source_manifest_git_head_mismatch_bundle', 'source_bundle_current_head_mismatch', 'source_manifest_git_head_mismatch_current', 'source_manifest_archive_file_missing', 'stage_receipt_missing', 'stage_receipt_invalid_or_empty', 'non_dry_run_command_placeholders_present', 'non_dry_run_command_arg_placeholder:--studio', 'non_dry_run_command_arg_placeholder:--teamspace', 'non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target', 'non_dry_run_command_identity_mode_not_exactly_one', 'active_lightning_claim_missing']` |
| `trained` | `[contest-CUDA]` | `lane_l5_v2_tt5l_sideinfo_effect_curve_trained_contest_cuda` | `l5-v2-tt5l-sideinfo-trained-cuda-20260517` | `False` | `['source_manifest:json_file_missing', 'source_manifest_missing', 'source_manifest_invalid_or_empty', 'source_manifest_artifact_paths_missing', 'source_manifest_git_head_mismatch_bundle', 'source_bundle_current_head_mismatch', 'source_manifest_git_head_mismatch_current', 'source_manifest_archive_file_missing', 'stage_receipt_missing', 'stage_receipt_invalid_or_empty', 'non_dry_run_command_placeholders_present', 'non_dry_run_command_arg_placeholder:--studio', 'non_dry_run_command_arg_placeholder:--teamspace', 'non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target', 'non_dry_run_command_identity_mode_not_exactly_one', 'active_lightning_claim_missing']` |
| `ablated` | `[contest-CPU]` | `lane_l5_v2_tt5l_sideinfo_effect_curve_ablated_contest_cpu` | `l5-v2-tt5l-sideinfo-ablated-cpu-20260517` | `False` | `['source_manifest:json_file_missing', 'source_manifest_missing', 'source_manifest_invalid_or_empty', 'source_manifest_artifact_paths_missing', 'source_manifest_git_head_mismatch_bundle', 'source_bundle_current_head_mismatch', 'source_manifest_git_head_mismatch_current', 'source_manifest_archive_file_missing', 'stage_receipt_missing', 'stage_receipt_invalid_or_empty', 'non_dry_run_command_placeholders_present', 'non_dry_run_command_arg_placeholder:--studio', 'non_dry_run_command_arg_placeholder:--teamspace', 'non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target', 'non_dry_run_command_identity_mode_not_exactly_one', 'active_lightning_claim_missing']` |
| `ablated` | `[contest-CUDA]` | `lane_l5_v2_tt5l_sideinfo_effect_curve_ablated_contest_cuda` | `l5-v2-tt5l-sideinfo-ablated-cuda-20260517` | `False` | `['source_manifest:json_file_missing', 'source_manifest_missing', 'source_manifest_invalid_or_empty', 'source_manifest_artifact_paths_missing', 'source_manifest_git_head_mismatch_bundle', 'source_bundle_current_head_mismatch', 'source_manifest_git_head_mismatch_current', 'source_manifest_archive_file_missing', 'stage_receipt_missing', 'stage_receipt_invalid_or_empty', 'non_dry_run_command_placeholders_present', 'non_dry_run_command_arg_placeholder:--studio', 'non_dry_run_command_arg_placeholder:--teamspace', 'non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target', 'non_dry_run_command_identity_mode_not_exactly_one', 'active_lightning_claim_missing']` |
