# No-Signal-Loss Canonicalization Plan - Worker C2 - 2026-05-08

generated_at_local: 2026-05-08
worker: Worker C2
branch: main
scope: plan-only canonicalization review after all-lanes preflight greened via
`.omx/research/untracked_source_dispositions_20260505_codex.json`

## Snapshot

This ledger does not stage, delete, revert, or commit any file. It records the
current review plan for the original 73 `track` dispositions added to
`.omx/research/untracked_source_dispositions_20260505_codex.json` relative to
HEAD by the green-preflight canonicalization pass. This ledger adds one
separate self-disposition entry for
`.omx/research/no_signal_loss_canonicalization_plan_20260508_worker_c2.md`;
that self-entry is not counted in the 73-path plan below.

Latest measured audit state:

- Manifest entries: 147 total.
- Manifest `track` entries: 143 total.
- Manifest delta relative to HEAD: 74 new entries: the original 73 planning
  entries plus this ledger's self-disposition; 0 existing entries changed.
- Current source-like untracked audit:
  - `untracked_source_like_count`: 73
  - `dispositioned_count`: 64
  - `undispositioned_count`: 9
  - `invalid_disposition_count`: 0
  - `resolved_tracked_disposition_count`: 83
- The original 73 track-disposition artifacts are still accounted for: 63
  remain live untracked, and 10 have resolved to tracked paths after index
  custody drift cleared. No disposition entry currently points at a missing or
  non-source path.

## Tracking Batches

Batch 1 - Research ledgers and report summaries - 22 paths:

- `.omx/research/all_lanes_preflight_blockers_20260508_codex.md`
- `.omx/research/arch_shrink_x0_4_lightning_poll_20260508_codex.md`
- `.omx/research/arch_shrink_x0_4_lightning_review_20260508_worker_a.md`
- `.omx/research/autopilot_evidence_semantics_review_20260508_worker_b.md`
- `.omx/research/cathedral_autopilot_lossy_coarsening_catalog_closure_20260508_worker_e1.md`
- `.omx/research/cross_paradigm_dispatch_readiness_review_20260508_worker_d.md`
- `.omx/research/hnerv_generated_schema_packet_scaffold_20260508_codex.md`
- `.omx/research/implementation_vs_model_verdict_chain_review_20260508_codex.md`
- `.omx/research/lightning_round3_harvest_and_runtime_failures_20260508_codex.md`
- `.omx/research/lossy_coarsening_exact_cuda_adversarial_review_20260508_worker_b.md`
- `.omx/research/lossy_coarsening_exact_cuda_result_review_20260508_codex.json`
- `.omx/research/lossy_falsification_scope_audit_20260508_codex.md`
- `.omx/research/lossy_int4_mixed_precision_coarsening_adversarial_review_20260508_codex.md`
- `.omx/research/monolithic_packet_bridge_review_20260508_worker_c.md`
- `.omx/research/monolithic_packet_candidate_bridge_20260508_codex.md`
- `.omx/research/monolithic_packet_closure_floor_gate_20260508_worker_m1.md`
- `.omx/research/omega_opt_anchor_discipline_20260508_codex.md`
- `.omx/research/paper_proxy_claim_language_audit_20260508_worker_d.md`
- `.omx/research/preflight_dirty_state_drift_review_20260508_worker_p1.md`
- `reports/cathedral_autopilot_catalog_updated_20260508.json`
- `reports/lossy_coarsening_exact_cuda_evidence_row_20260508.json`
- `reports/omega_opt_linear_stack_packet_20260508.json`

Review rule: track these together as durable control-plane summaries. Keep raw
provider logs and rebuildable experiment trees ignored unless separately
manifested.

Batch 2 - TAC reusable implementation surfaces - 9 paths:

- `src/tac/codec/__init__.py`
- `src/tac/codec/syndrome_trellis_codec.py`
- `src/tac/deploy/lightning/round3_harvest.py`
- `src/tac/hnerv_generated_schema_packet.py`
- `src/tac/monolithic_codec_op_replacement.py`
- `src/tac/monolithic_packet_candidate.py`
- `src/tac/monolithic_packet_closure_gate.py`
- `src/tac/omega_opt_claims.py`
- `src/tac/omega_opt_linear_stack_packet.py`

Review rule: inspect imports and public preflight/tool callsites before staging.
`src/tac/codec/syndrome_trellis_codec.py` is already a tracked path in the
latest snapshot, so it should be reviewed as a normal tracked modification, not
as a new untracked file.

Batch 3 - TAC regression tests - 21 paths:

- `src/tac/tests/test_build_hnerv_generated_schema_candidate.py`
- `src/tac/tests/test_build_monolithic_runtime_consumption_proof.py`
- `src/tac/tests/test_build_result_review_packet.py`
- `src/tac/tests/test_export_active_lane_claim_json.py`
- `src/tac/tests/test_hnerv_generated_schema_packet.py`
- `src/tac/tests/test_monolithic_codec_op_replacement.py`
- `src/tac/tests/test_monolithic_packet_candidate.py`
- `src/tac/tests/test_monolithic_packet_closure_gate.py`
- `src/tac/tests/test_omega_opt_anchor_discipline_tool.py`
- `src/tac/tests/test_omega_opt_claims.py`
- `src/tac/tests/test_omega_opt_linear_stack_packet.py`
- `src/tac/tests/test_pr101_cross_paradigm_hstack_vstack.py`
- `src/tac/tests/test_pr101_lossy_proxy_guardrails.py`
- `src/tac/tests/test_pr101_omega_opt_admm_x_lossy_coarsening.py`
- `src/tac/tests/test_pr101_omega_opt_joint_admm_allocation.py`
- `src/tac/tests/test_pr101_omega_opt_uniward_weighted.py`
- `src/tac/tests/test_pr_alpha_mask_stc_empirical.py`
- `src/tac/tests/test_preflight_harden_2026_05_08_checks.py`
- `src/tac/tests/test_prove_hnerv_generated_schema_runtime_packet.py`
- `src/tac/tests/test_run_monolithic_candidate_preflight.py`
- `src/tac/tests/test_syndrome_trellis_codec.py`

Review rule: stage with the implementation or tool batch they protect. The
PR101 omega/STC/preflight-hardening tests are already tracked paths in the
latest snapshot and should be reviewed for tracked-file intent before any
candidate commit boundary is chosen.

Batch 4 - Operator-visible tools, builders, scanners, and proof CLIs - 19 paths:

- `tools/build_hnerv_generated_schema_candidate.py`
- `tools/build_monolithic_codec_op_replacement_manifest.py`
- `tools/build_monolithic_runtime_consumption_proof.py`
- `tools/build_monolithic_stack_candidate.py`
- `tools/build_result_review_packet.py`
- `tools/check_admm_naming_matches_iterative_consensus_implementation.py`
- `tools/check_build_manifest_archive_custody_clean.py`
- `tools/check_encoder_decoder_dequantization_roundtrip_tested.py`
- `tools/check_evidence_row_archive_bytes_has_provenance.py`
- `tools/check_inflate_wire_format_no_dead_bytes.py`
- `tools/check_monolithic_packet_closure_gate.py`
- `tools/check_omega_opt_anchor_discipline.py`
- `tools/check_omega_opt_linear_stack_packet.py`
- `tools/check_pr101_tools_torch_load_allowlist.py`
- `tools/check_predispatch_retired_config_warning.py`
- `tools/export_active_lane_claim_json.py`
- `tools/materialize_omega_opt_linear_stack_packet.py`
- `tools/prove_hnerv_generated_schema_runtime_packet.py`
- `tools/run_monolithic_candidate_preflight.py`

Review rule: confirm each promoted guard is wired into an operator-visible
surface such as `tools/all_lanes_preflight.py`, `src/tac/preflight.py`, a
documented runbook, or a dated control ledger.

Batch 5 - Lane-specific restored tools - 2 paths:

- `tools/pr101_omega_opt_uniward_weighted_allocation.py`
- `tools/pr_alpha_mask_stc_empirical.py`

Review rule: these are now tracked paths in the latest snapshot. Review them as
tracked edits or restored tracked files, and keep their commit boundary near
the matching tests in Batch 3.

## Current Drift To Resolve Before Green

The latest audit is no longer green because 9 source-like untracked paths lack
dispositions. These appear to be new worker files after the green preflight
snapshot:

- `src/tac/tests/preflight/__init__.py`
- `src/tac/tests/preflight/test_check_b1_encoder_decoder_roundtrip.py`
- `src/tac/tests/preflight/test_check_b2_evidence_archive_bytes_provenance.py`
- `src/tac/tests/preflight/test_check_b3_build_manifest_archive_custody.py`
- `src/tac/tests/preflight/test_check_b4_admm_naming_impl.py`
- `src/tac/tests/preflight/test_check_b5_inflate_dead_bytes.py`
- `src/tac/tests/preflight/test_check_b6_retired_config_redispatch.py`
- `src/tac/tests/preflight/test_check_b7_paper_research_score_lane_tag.py`
- `src/tac/tests/preflight/test_check_b8_pr101_torch_load_allowlist.py`

Recommended disposition: add explicit `track` dispositions after owner review.
The nine `src/tac/tests/preflight/*` files are focused guard regression tests.
An intermediate status snapshot also showed
`tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py` as
undispositioned, but the latest `git status` and `git ls-files` show it is a
tracked clean path and not a current untracked-source blocker.

Original 73 paths that are no longer live untracked but remain accounted for as
tracked/resolved:

- `src/tac/codec/syndrome_trellis_codec.py`
- `src/tac/tests/test_pr101_cross_paradigm_hstack_vstack.py`
- `src/tac/tests/test_pr101_omega_opt_admm_x_lossy_coarsening.py`
- `src/tac/tests/test_pr101_omega_opt_joint_admm_allocation.py`
- `src/tac/tests/test_pr101_omega_opt_uniward_weighted.py`
- `src/tac/tests/test_pr_alpha_mask_stc_empirical.py`
- `src/tac/tests/test_preflight_harden_2026_05_08_checks.py`
- `src/tac/tests/test_syndrome_trellis_codec.py`
- `tools/pr101_omega_opt_uniward_weighted_allocation.py`
- `tools/pr_alpha_mask_stc_empirical.py`

## Review Sequence

1. Resolve the 9 undispositioned source-like paths in the manifest, or record
   a non-track disposition with rationale if owner review rejects promotion.
2. Re-run `tools/audit_untracked_source_artifacts.py` against the manifest and
   require `ready_for_no_signal_loss_canonicalization=true`.
3. Re-run the release/index split and orphan canonicalization audits before any
   actual staging, because this worktree has concurrent partner WIP.
4. Stage only by the batches above, keeping implementation, tests, tools, and
   ledgers in reviewable commit boundaries.
5. Do not stage raw `experiments/results/*`, provider logs, extracted payloads,
   or private state unless a separate manifest-led promotion decision is made.
