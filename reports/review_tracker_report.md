# Code Review Tracker Report — 2026-05-23 20:05 UTC

## Summary

- **Total entities**: 75247
- **Reviewed**: 65029 (86%)
- **Unreviewed**: 10218
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

- `tac.tests.test_optimizer_exact_readiness::test_active_floor_score_tracks_score_frontier_not_rate_only_anchor` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_load_parallel_dispatch_tool` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_write_json` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_write_archive` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_make_submission` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_make_queue` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_mark_queue_row_as_inverse_scorer_chain` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_hdm8_selector_gate` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_selector_cuda_transfer_calibration` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_mark_submission_as_hdm8_selector` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_write_pr101_runtime_proof` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::_add_required_runtime_proof_fields` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::test_promotes_byte_closed_candidate_without_score_claim` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::test_refuses_inverse_scorer_chain_without_strict_full_frame_parity` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::test_refuses_inverse_scorer_chain_without_exact_auth_score_boundary` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::test_refuses_inverse_scorer_chain_with_self_asserted_unbacked_parity` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::test_refuses_inverse_scorer_chain_with_truthy_false_authority_fields` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::test_promotes_inverse_scorer_chain_only_after_parity_and_auth_boundary` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::test_refuses_hdm8_selector_without_passing_cuda_component_gate` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
- `tac.tests.test_optimizer_exact_readiness::test_promotes_hdm8_selector_after_passing_cuda_component_gate` — marked_reviewed by codex (codex_inverse_queue_storage_review_pass1)
