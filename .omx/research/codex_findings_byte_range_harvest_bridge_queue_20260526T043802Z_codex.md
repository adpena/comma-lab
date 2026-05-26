# Byte-Range Harvest Bridge Queue Closure

UTC: 2026-05-26T04:38:02Z

## Finding

The byte-range entropy-recode materializer was queue-executable but its signal still had two integration leaks:

- harvested `optimizer_candidate_queue_v1` source queues did not carry top-level false-authority fields, so queue postconditions could not assert the source queue safely;
- exact-readiness did not recognize `byte_range_entropy_recode_receiver_proof_v1`, so a valid local receiver/runtime proof was reported as `runtime_consumption_proof_schema_unsupported`.

Both are now fixed. The operation-chain compiler queue runs:

1. operation-chain stage plan;
2. byte-range stage inputs;
3. byte-range entropy-recode chain;
4. materializer-chain harvest into optimizer source queue;
5. submission runtime closure;
6. exact-readiness bridge.

The chain command is now overwrite-capable so a queue rewind can reproduce artifacts in place.

## Live Smoke

Queue artifact:

`.omx/research/frontier_rate_attack_feedback_cycle_20260526T_byte_range_harvest_bridge/initial_refresh/operation_chain_compiler_queue.json`

Smoke summary:

`.omx/research/frontier_rate_attack_feedback_cycle_20260526T_byte_range_harvest_bridge/initial_refresh/byte_range_harvest_bridge_smoke_summary.json`

Observed live result:

- `step_count`: 6
- worker status: 6 succeeded
- target kind: `byte_range_entropy_recode_v1`
- materializer id: `byte_range_entropy_recode_adapter`
- receiver contract id: `byte_range_entropy_recode_receiver.v1`
- receiver contract kind: `archive_charged_byte_range_entropy_recode`
- source archive bytes: 178223
- candidate archive bytes: 178207
- realized saved bytes: 16
- receiver contract satisfied: true
- runtime adapter ready: true
- harvest accepted manifests: 1
- exact-readiness ready candidates: 0
- exact-readiness blocked candidates: 1

Remaining blockers are concrete and appropriate:

- `candidate_inflate_output_parity_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `full_frame_render_output_parity_missing`
- `shell_inflate_output_parity_missing`

The previous false blocker `runtime_consumption_proof_schema_unsupported` is gone.

## Correction-Budget Semantics

The rate win is now visible to the receiver-closed correction-budget scanner, but it is not released:

- saved bytes at risk: 16
- `receiver_closed`: false
- `release_to_targeted_correction_planning`: false
- `ready_for_budget_spend`: false

This preserves the intended discipline: byte savings can become targeted SegNet/PoseNet correction budget only after receiver parity and exact-readiness handoff gates clear.

## Verification

- `ruff` on touched scheduler/materializer/exact-readiness/test files: passed
- focused regression slice: 145 passed
- byte-shaving queue/runner slice: 160 passed
- live operation-chain queue worker: 6/6 steps succeeded
- `tools/lane_maturity.py validate`: 1378 lanes clean

Global `tools/review_tracker.py policy-check` is not a useful scoped gate in the current repo state: it reports 11560 pre-existing unrelated review violations. Touched files were marked reviewed after the focused diff and regression pass.

## Subagent Note

An xhigh sidecar spawn was attempted for this tranche, but the agent thread limit was already reached. No sidecar edits were made.

