# Codex Findings: Operation Chain Stage-Plan Queue

UTC: 2026-05-26T03:53:20Z

## Verdict

The multisurface chain work order was discoverable, but still required an
operator or agent to manually open the work-order JSON and decide how to stage
the compiler work. This landing turns that handoff into a queue-owned actuator
surface:

`operation_chain_compiler_work_orders.json -> operation_chain_compiler_queue.json -> stage_plan.json`

The stage plan is still planning-only. It does not claim score, promotion,
rank/kill, dispatch readiness, charged-bit change, or exact-eval authority.

## Implemented

- Added `build_frontier_operation_chain_compiler_stage_plan(...)`.
  - Converts a single chain work order into explicit stage rows.
  - Preserves missing contracts from bridge blockers.
  - Emits queue handoffs to:
    - `byte_shaving_campaign_queue`
    - `frontier_receiver_repair_queue`
    - `frontier_targeted_component_correction_queue`
- Added `build_frontier_operation_chain_compiler_queue(...)`.
  - Emits local-IO queue rows that materialize stage plans.
  - Forces `execution_ready=false` and `budget_spend_allowed=false`.
- Added `tools/build_frontier_operation_chain_stage_plan.py`.
  - Reads `operation_chain_compiler_work_orders.json`.
  - Writes one `frontier_rate_attack_operation_chain_compiler_stage_plan.v1`.
- Wired `write_frontier_refresh_artifacts(...)` to emit
  `operation_chain_compiler_queue.json`.
- Exported the new builders from `comma_lab.scheduler`.

## Live Evidence

Generated root:
`.omx/research/frontier_rate_attack_feedback_cycle_20260526T_chain_stage_plan_queue/`

New queue artifact:
`.omx/research/frontier_rate_attack_feedback_cycle_20260526T_chain_stage_plan_queue/initial_refresh/operation_chain_compiler_queue.json`

Queue validation:

- queue id: `frontier_rate_attack_feedback_cycle_chain_stage_plan_queue_20260526_chain_compiler`
- experiment count: 1
- step count: 1
- valid: true

Executed queue step output:
`experiments/results/frontier_operation_chain_compiler/frontier_rate_attack_feedback_cycle_chain_stage_plan_queue_20260526_chain_compiler/chain_registered_multisurface_materializer_program/stage_plan.json`

Durable research mirror:
`.omx/research/frontier_rate_attack_feedback_cycle_20260526T_chain_stage_plan_queue/initial_refresh/operation_chain_stage_plan_smoke.json`

Stage plan summary:

- schema: `frontier_rate_attack_operation_chain_compiler_stage_plan.v1`
- source operation: `chain_registered_multisurface_materializer_program`
- stage count: 4
- execution ready: false
- missing contracts:
  - `payload_grammar_schema_manifest`
  - `archive_section_header_elision_contract`
  - `archive_section_order_independence_contract`
  - `tensor_sensitivity_rank_quant_prune_contract`
  - `shared_codebook_dictionary_contract`
  - `single_composed_receiver_runtime_consumption_proof`
  - `chain_exact_readiness_handoff_after_composition`

Stages:

1. `payload_grammar_and_entropy`
   - target: `byte_range_entropy_recode_v1`
   - requires: `schema_manifest`, `beam_probe_reports`, `source_runtime_dir`
2. `archive_section_receiver_contracts`
   - targets: `archive_section_header_elide_v1`,
     `archive_section_reorder_v1`, `archive_section_proceduralize_v1`
   - requires: `header_elision_contract`, `section_order_contract`,
     `runtime_consumption_proof`
3. `tensor_scorer_sensitive_layout`
   - targets: `tensor_quantize_v1`, `tensor_prune_v1`,
     `tensor_shared_codebook_v1`
   - requires: `component_sensitivity_rows`,
     `receiver_exact_reconstruction_contract`
4. `packet_member_lookup_and_high_level_action_sets`
   - targets: `packet_member_reorder_v1`,
     `inverse_steganalysis_high_level_operation_set_v1`
   - requires: `member_lookup_proof`,
     `inverse_scorer_action_surface_binding`

## Verification

- `ruff check` on touched scheduler/test/tool files: pass.
- Focused feedback-cycle regressions: 3 passed.
- Wider feedback/context/queue regression:
  `pytest src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  -> 132 passed.
- `tools/experiment_queue.py --queue .../operation_chain_compiler_queue.json validate`
  -> valid.
- Stage-plan CLI smoke wrote the expected stage-plan JSON.

## Remaining Work

The next queue-owned tranche should let one stage become more than a plan. The
highest-leverage first target is byte-range entropy recode, because prior
receiver-proof work already exists around `pr103_lcac_runtime_adapter.py` and
`byte_range_entropy_recode_receiver_proof_v1`. The next step is to bind that
existing receiver-proof builder into the stage-plan queue so stage 1 can emit
`schema_manifest`, `beam_probe_reports`, and `source_runtime_dir` artifacts
rather than only blockers.
