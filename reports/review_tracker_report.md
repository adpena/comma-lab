# Code Review Tracker Report — 2026-07-13 00:09 UTC

## Summary

- **Total entities**: 123620
- **Reviewed**: 104536 (85%)
- **Unreviewed**: 18918
- **Stale**: 166
- **Needs fix**: 0

## Priority Review Queue (by complexity)

| Entity | Type | Lines | Complexity | Status | File |
|--------|------|-------|------------|--------|------|
| `_stamp_expected_source_sha256` | function | 6 | 1 | unreviewed | artifact_retention.py |
| `sync_repo` | function | 7 | 1 | unreviewed | lossless_review_tracker.py |
| `run` | function | 2 | 1 | unreviewed | bootstrap.py |
| `install_payload_bytes` | function | 2 | 1 | unreviewed | install.py |
| `install_payload_manifest` | function | 5 | 1 | unreviewed | install.py |
| `install_payload_paths` | function | 15 | 1 | unreviewed | install.py |
| `install_submission` | function | 22 | 1 | unreviewed | install.py |
| `_file_record` | function | 14 | 1 | unreviewed | local_submission_replay.py |
| `bootstrap_upstream` | function | 19 | 1 | unreviewed | bootstrap.py |
| `_is_pid_alive` | function | 23 | 1 | unreviewed | lock.py |
| `_lock_dir` | function | 3 | 1 | unreviewed | lock.py |
| `submission_lock` | function | 60 | 1 | unreviewed | lock.py |
| `_atomic_write_text` | function | 10 | 1 | unreviewed | lossless_review_tracker.py |
| `_load_json` | function | 5 | 1 | unreviewed | lossless_review_tracker.py |
| `_lossless_entity_key` | function | 15 | 1 | unreviewed | lossless_review_tracker.py |
| `_project_payload` | function | 17 | 1 | unreviewed | lossless_review_tracker.py |
| `_render_payload` | function | 2 | 1 | unreviewed | lossless_review_tracker.py |
| `canonical_tracker_path` | function | 2 | 1 | unreviewed | lossless_review_tracker.py |
| `doctor_repo` | function | 15 | 1 | unreviewed | lossless_review_tracker.py |
| `load_global_tracker` | function | 6 | 1 | unreviewed | lossless_review_tracker.py |
| `project_tracker` | function | 2 | 1 | unreviewed | lossless_review_tracker.py |
| `scan_repo` | function | 4 | 1 | unreviewed | lossless_review_tracker.py |
| `status_payload` | function | 17 | 1 | unreviewed | lossless_review_tracker.py |
| `_normalize_lossless_result_payload` | function | 6 | 1 | unreviewed | lossless_state_sync.py |
| `_atomic_write_text` | function | 10 | 1 | unreviewed | lossless_state_sync.py |
| `_dedupe_rows` | function | 13 | 1 | unreviewed | lossless_state_sync.py |
| `_load_json` | function | 5 | 1 | unreviewed | lossless_state_sync.py |
| `_lossless_result_type` | function | 3 | 1 | unreviewed | lossless_state_sync.py |
| `_lossless_symbol` | function | 8 | 1 | unreviewed | lossless_state_sync.py |
| `_selected_replay_env` | function | 3 | 1 | unreviewed | local_submission_replay.py |

## Recent Review Activity

- `tac.tests.test_probe_jrd_pr110_coefficient_prefix::test_refuse_unsafe_scope_requires_results_child` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tac.tests.test_probe_jrd_pr110_coefficient_prefix::test_section_summary_separates_last_safe_from_best_rate` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tac.tests.test_probe_jrd_pr110_coefficient_prefix::test_resume_fails_closed_on_fingerprint_change` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tac.tests.test_probe_jrd_pr110_coefficient_prefix::test_parse_args_bounds_pair_counts` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tac.tests.test_probe_jrd_pr110_coefficient_prefix::test_controls_require_stable_positive_and_both_negative_component_changes` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tac.tests.test_jrd_pr110_coefficient_prefix::test_submission_byte_map_inverse_covers_all_int8_values` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tac.tests.test_jrd_pr110_coefficient_prefix::test_unknown_byte_map_refuses` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tac.tests.test_jrd_pr110_coefficient_prefix::test_real_archive_derives_28_sections_and_noop_is_byte_exact` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tac.tests.test_jrd_pr110_coefficient_prefix::test_real_single_tensor_prefix_preserves_every_other_grammar_section` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tac.tests.test_jrd_pr110_coefficient_prefix::test_replacement_refuses_wrong_dtype_shape_and_forged_section` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tools.probe_jrd_pr110_coefficient_prefix::sha256_bytes` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tools.probe_jrd_pr110_coefficient_prefix::sha256_file` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tools.probe_jrd_pr110_coefficient_prefix::validate_controls` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tools.probe_jrd_pr110_coefficient_prefix::ExactLocalMeter` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tools.probe_jrd_pr110_coefficient_prefix::ExactLocalMeter.__init__` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tools.probe_jrd_pr110_coefficient_prefix::_prefix_row` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tools.probe_jrd_pr110_coefficient_prefix::_section_summary` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tools.probe_jrd_pr110_coefficient_prefix::run` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tools.probe_jrd_pr110_coefficient_prefix::parse_args` — marked_reviewed by fresh_context_triplicate (clean_3)
- `tools.probe_jrd_pr110_coefficient_prefix::main` — marked_reviewed by fresh_context_triplicate (clean_3)
