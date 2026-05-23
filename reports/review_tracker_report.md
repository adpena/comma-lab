# Code Review Tracker Report — 2026-05-23 20:19 UTC

## Summary

- **Total entities**: 75250
- **Reviewed**: 65036 (86%)
- **Unreviewed**: 10214
- **Stale**: 0
- **Needs fix**: 0

## Priority Review Queue (by complexity)

| Entity | Type | Lines | Complexity | Status | File |
|--------|------|-------|------------|--------|------|
| `InverseScorerCellChainError` | class | 2 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `_resolve_optional_dir` | function | 7 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_chain_manifest` | function | 123 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `_next_required_gates` | function | 12 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `_prepare_new_output_dir` | function | 4 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `_write_json` | function | 2 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `_write_failure_manifest` | function | 23 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `_file_record` | function | 6 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `build_inverse_scorer_cell_candidate_chain` | function | 129 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `_mapping` | function | 2 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `_string_list` | function | 2 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `_ordered_unique` | function | 8 | 1 | unreviewed | inverse_scorer_cell_chain.py |
| `InverseScorerCellInflateParityError` | class | 2 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `build_inverse_scorer_cell_inflate_parity_probe` | function | 112 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `build_inverse_scorer_cell_inflate_parity_probe_from_archives` | function | 160 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `verify_inverse_scorer_cell_inflate_parity_probe` | function | 96 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_output_tree_record` | function | 41 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_missing_tree_record` | function | 10 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_file_map` | function | 17 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_load_candidate` | function | 22 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_load_optional_mapping` | function | 13 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_archive_record` | function | 10 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_descriptor_record` | function | 12 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_preflight_blocked_inflate_run` | function | 20 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_resolve_existing_path` | function | 9 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_match_text` | function | 10 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_canonical_tree_sha` | function | 10 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_blocked_archive_parity_probe` | function | 54 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_run_inflate` | function | 69 | 1 | unreviewed | inverse_scorer_cell_inflate_parity.py |
| `_repo_path` | function | 2 | 1 | unreviewed | inverse_scorer_cell_chain.py |

## Recent Review Activity

- `tac.tests.test_recover_modal_auth_eval_tool::_load_tool` — marked_reviewed by codex (ias1_recovery_idempotence_review)
- `tac.tests.test_recover_modal_auth_eval_tool::_write_auth_eval` — marked_reviewed by codex (ias1_recovery_idempotence_review)
- `tac.tests.test_recover_modal_auth_eval_tool::test_terminal_status_uses_exact_readiness_cuda_prefix` — marked_reviewed by codex (ias1_recovery_idempotence_review)
- `tac.tests.test_recover_modal_auth_eval_tool::test_terminal_notes_include_exact_custody_fields` — marked_reviewed by codex (ias1_recovery_idempotence_review)
- `tac.tests.test_recover_modal_auth_eval_tool::test_auth_eval_artifact_path_accepts_adjudicated_fallback` — marked_reviewed by codex (ias1_recovery_idempotence_review)
- `tac.tests.test_recover_modal_auth_eval_tool::test_maybe_update_posterior_routes_auth_eval_artifact` — marked_reviewed by codex (ias1_recovery_idempotence_review)
- `tac.tests.test_recover_modal_auth_eval_tool::test_main_requires_auditable_no_close_reason` — marked_reviewed by codex (ias1_recovery_idempotence_review)
- `tac.tests.test_recover_modal_auth_eval_tool::test_main_fails_loud_when_terminal_metadata_lacks_claim_fields` — marked_reviewed by codex (ias1_recovery_idempotence_review)
- `tac.tests.test_recover_modal_auth_eval_tool::test_main_allows_no_close_only_with_auditable_reason` — marked_reviewed by codex (ias1_recovery_idempotence_review)
- `tac.tests.test_recover_modal_auth_eval_tool::test_main_skips_duplicate_terminal_recovery_without_posterior_touch` — marked_reviewed by codex (ias1_recovery_idempotence_review)
- `tac.tests.test_byte_shaving_campaign_queue::_byte_range_entropy_plan` — marked_reviewed by codex (ias1_scheduler_storage_preflight_review)
- `tac.tests.test_byte_shaving_campaign_queue::_inverse_surface_plan` — marked_reviewed by codex (ias1_scheduler_storage_preflight_review)
- `tac.tests.test_byte_shaving_campaign_queue::_inverse_cell_candidate_plan` — marked_reviewed by codex (ias1_scheduler_storage_preflight_review)
- `tac.tests.test_byte_shaving_campaign_queue::test_byte_shaving_materializer_registry_exposes_dqs1_and_byte_range_contracts` — marked_reviewed by codex (ias1_scheduler_storage_preflight_review)
- `tac.tests.test_byte_shaving_campaign_queue::test_byte_shaving_materializer_registry_refuses_implicit_dqs1_pair_drop` — marked_reviewed by codex (ias1_scheduler_storage_preflight_review)
- `tac.tests.test_byte_shaving_campaign_queue::test_byte_shaving_materializer_registry_allows_explicit_dqs1_target_kind` — marked_reviewed by codex (ias1_scheduler_storage_preflight_review)
- `tac.tests.test_byte_shaving_campaign_queue::test_byte_shaving_materializer_registry_allows_inverse_action_probe_target_kind` — marked_reviewed by codex (ias1_scheduler_storage_preflight_review)
- `tac.tests.test_byte_shaving_campaign_queue::test_compile_dqs1_byte_shaving_plan_classifies_byte_range_entropy_contract_gap` — marked_reviewed by codex (ias1_scheduler_storage_preflight_review)
- `tac.tests.test_byte_shaving_campaign_queue::test_compile_dqs1_byte_shaving_plan_suggests_byte_range_entropy_target_kind` — marked_reviewed by codex (ias1_scheduler_storage_preflight_review)
- `tac.tests.test_byte_shaving_campaign_queue::test_materializer_work_queue_builds_byte_range_chain_command` — marked_reviewed by codex (ias1_scheduler_storage_preflight_review)
