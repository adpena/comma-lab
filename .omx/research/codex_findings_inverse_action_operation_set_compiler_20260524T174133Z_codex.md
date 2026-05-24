# Codex Findings: Inverse-Action Operation-Set Compiler

- utc: 2026-05-24T17:41:33Z
- lane_id: codex_inverse_action_operation_set_compiler_20260524
- role: Codex executor / adversarial reviewer
- authority: planning-only; no score claim; no promotion; no dispatch

## Landing

Added the first deterministic compiler path for inverse-action water-bucket cells that do not already carry source-provenance operation sets.

The action functional now preserves an optional `operation_set_compiler` mapping from normalized inverse-steganalysis atoms into full action cells. The selected water-bucket row remains compact; the compiler hint stays on the full cell and is consumed by `build_signal_surface_from_inverse_action_functional`.

Supported compiler targets in this slice:

- `archive_section_entropy_recode_v1`
- `packet_member_recompress_v1`
- `tensor_factorize_v1`

When the full action cell carries explicit supported compiler hints, the byte-shaving bridge emits concrete family operations with `actuation_mode="compiled_operation_set"`. The existing plan path then emits PacketIR operation sets, and the standalone materialization bridge marks the corresponding portfolio row queue-consumable. Unsupported compiler hints fail closed back to `high_level_operation_compiler_required`.

## Safeguards

This remains planning-only. The compiled operations preserve false-authority fields and still carry blockers for materializer contexts, runtime-consumption proof, and exact auth eval. No MLX/proxy/local row becomes score, promotion, rank/kill, dispatch, or exact-eval authority.

The patch also fixes a PacketIR handoff bug: bounded permutation order records were dropping operation `params` before PacketIR lowering. PacketIR now restores full selected-operation details while preserving the chosen order.

## Verification

Passed:

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_inverse_steganalysis_acquisition.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_packet_ir_operation_set_lowers_to_materializer_backlog_rows -q`
- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py::test_final_byte_context_compiler_covers_inverse_scorer_cell_candidate src/tac/tests/test_final_byte_operation_contexts.py::test_final_byte_context_compiler_turns_unsupported_rows_into_blocked_contexts -q`
- `.venv/bin/python -m ruff check src/tac/optimization/byte_shaving_campaign.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_inverse_steganalysis_acquisition.py`
- `git diff --check -- src/tac/optimization/byte_shaving_campaign.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_inverse_steganalysis_acquisition.py .omx/state/lane_registry.json .omx/state/lane_maturity_audit.log`

## Remaining Gap

This is a narrow compiler: it compiles explicit hints into already-supported final-byte families. The next frontier step is not another descriptor. The next step is to have MLX/Metal scorer-response acquisition emit these compiler hints directly from hydrated contest-video/scorer/rate coordinates, then let the queue-owned campaign runner materialize, verify, harvest, learn, and replan.
