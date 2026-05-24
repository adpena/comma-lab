# Codex Findings: Inverse-Action Materialization Bridge

- utc: 2026-05-24T17:32:49Z
- lane_id: codex_inverse_action_materialization_bridge_20260524
- role: Codex executor / adversarial reviewer
- scope: inverse-steganalysis water-bucket portfolio observability and queue handoff
- authority: planning-only; no score claim; no promotion; no dispatch

## Landing

Added a standalone typed bridge artifact:

- schema: `inverse_steganalysis_water_bucket_materialization_bridge.v1`
- builder: `tac.optimization.byte_shaving_campaign.build_inverse_action_materialization_bridge`
- CLI output: `tools/plan_byte_shaving_campaign.py --inverse-action-materialization-bridge-out`

The bridge extracts inverse-action water-bucket portfolio rows from a byte-shaving campaign plan, links each portfolio row to matching PacketIR operation sets when source provenance already rehydrates into concrete family operations, and explicitly counts blocked compiler-required rows.

## Why This Matters

The system already had a water-bucket portfolio embedded inside the plan, but it was too easy for operators and downstream automation to miss the decisive distinction:

- `source_provenance_operation_set`: can already lower through existing PacketIR/materializer queue machinery.
- `high_level_operation_compiler_required`: selected inverse-action cell is still a compiler gap, not a byte-changing final operation.
- `leaf_cell_candidate_explicit_opt_in`: probe-only diagnostic path, not a portfolio actuator.

The bridge makes that distinction machine-readable through:

- `portfolio_row_bridge_links`
- `queue_consumable_portfolio_row_count`
- `queue_consumable_packet_ir_operation_set_ids`
- `high_level_operation_compiler_required_count`
- `queue_consumption.next_gate`

## Safeguards

The bridge recursively refuses truthy authority fields from the source plan and applies the canonical proxy evidence boundary to its own payload. It preserves false values for score, promotion, rank/kill, dispatch, exact-eval readiness, and GPU launch authority.

The bridge does not claim that bare water-bucket cells are executable. When PacketIR is absent it records `inverse_action_operation_set_compiler` as the next gate and carries explicit blockers.

## Verification

Passed:

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py::test_builder_merges_inverse_action_functional_into_mixed_signal_surface src/tac/tests/test_byte_shaving_campaign_queue.py::test_packet_ir_operation_set_lowers_to_materializer_backlog_rows src/tac/tests/test_final_byte_operation_contexts.py::test_final_byte_context_compiler_covers_inverse_scorer_cell_candidate src/tac/tests/test_final_byte_operation_contexts.py::test_final_byte_context_compiler_turns_unsupported_rows_into_blocked_contexts -q`
- `.venv/bin/python -m ruff check src/tac/optimization/byte_shaving_campaign.py tools/plan_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign.py`
- `git diff --check -- src/tac/optimization/byte_shaving_campaign.py tools/plan_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign.py .omx/state/lane_registry.json .omx/state/lane_maturity_audit.log`

Note: the worktree also contained unrelated partner/user changes in HNeRV low-level ZIP hardening and scheduler materializer tests. They were used only as current-tree compatibility smoke context and are not part of this bridge landing.

## Next 4-Week Tranche Delta

Week 1 P0: implement the actual `inverse_action_operation_set_compiler` for non-provenance water-bucket cells. It should emit concrete family operation sets and PacketIR rows instead of leaf descriptors.

Week 2 P0: wire compiler output through `build_byte_shaving_campaign_queue.py`, `build_final_byte_operation_contexts.py`, and the materializer work queue so a selected water-bucket portfolio row becomes byte-changing work without manual row surgery.

Week 3 P0: run local MLX/Metal/Accelerate scorer-response acquisition batches that emit source-provenance operation sets directly, then regenerate the bridge and queue to prove queue-consumable rows increase.

Week 4 P0: close exact-readiness dry-run loops for the best local byte-changing outputs while preserving axis separation and no-score authority until contest CPU/CUDA auth eval.

## Outstanding Gap

The remaining hard gap is not bridge visibility; it is compilation. Bare inverse-action selected cells still need a deterministic family-aware compiler that turns hydrated scorer/action-surface coordinates into concrete operation families over HNeRV, HNeRV boltons, NeRV-family variants, and non-NeRV packet structures.
