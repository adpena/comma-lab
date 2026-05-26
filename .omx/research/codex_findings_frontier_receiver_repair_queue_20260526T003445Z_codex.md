# Frontier Receiver Repair Queue Wire-In

Generated at: 2026-05-26T00:34:45Z
Lane: lane_frontier_receiver_repair_queue_20260526

## Verdict

The frontier operation portfolio is no longer only metadata on DQS1 follow-up
rows. The top receiver/exact-readiness blockers now compile into a bounded,
source-diversified `experiment_queue.v1` receiver-repair queue, and each queued
step emits a typed work order that binds repair family, bridge reports when
available, candidate ids, exact-ready handoff paths, command hints, and the
targeted-correction budget spend gate.

The 794 materializer saved bytes and 28 local drop saved bytes remain
false-authority planning budget. They cannot fund SegNet/PoseNet repair search
until the receiver runtime/parity work order clears runtime consumption.

## What Changed

- Added `frontier_rate_attack_receiver_repair_work_order.v1`.
- Preserved bridge report paths, candidate ids, and blocker samples on receiver
  repair backlog rows.
- Added `build_frontier_receiver_repair_work_order(...)` and
  `build_frontier_receiver_repair_queue(...)`.
- Made receiver-repair queue selection source-diversified first, then
  family-diversified, so the default queue covers multiple materializer/receiver
  sources instead of spending every slot on one blocker cluster.
- Added `tools/build_frontier_receiver_repair_work_order.py`.
- Wired `receiver_repair_queue.json` into both feedback refresh artifact
  writers and operator validation commands.
- Exported the new helpers through `comma_lab.scheduler`.
- Added regression coverage that validates queue emission, deterministic
  retry-safe work-order generation, false-authority preservation, and the
  correction-budget spend gate.

## Live Artifact

Canonical explicit-input artifact regenerated:

`.omx/research/frontier_operation_portfolio_explicit_receiver_repair_20260526T003426Z/`

Key outputs:

- `operation_portfolio.json`: 32 operation rows, 5 queue-executable rows, 14
  follow-up-signal rows.
- `receiver_repair_backlog.json`: 173 repair rows, 119 queue-actionable repair
  rows.
- `receiver_repair_queue.json`: 4 receiver repair experiments, 4 work-order
  steps, valid. The selected sources are the DFL1/merge/header-elide chain,
  standalone DFL1, packet-member merge, and ZIP header elide.
- `dqs1_followup_queue.json`: 4 DQS1 experiments, 28 steps, valid.

An adjacent auto-refresh artifact also exists at
`.omx/research/frontier_operation_portfolio_20260526T003445Z/`. It is valid,
but was generated without the explicit materializer feedback paths, so it is a
fallback/no-materializer-budget view: 36 receiver-repair rows, 30
queue-actionable rows, `materializer_rate_positive_saved_bytes_total=0`, and
`local_drop_saved_bytes_total=28`.

Top repair families:

1. full_frame_inflate_parity_repair
2. runtime_consumption_proof_repair

Current queue work orders:

- `chain_dfl1_merge_header_elide_minimal_envelope`: full-frame DFL1 parity
  repair hint.
- `materializer_renderer_payload_dfl1_v1`: standalone full-frame DFL1 parity
  repair hint.
- `materializer_packet_member_merge_v1`: bridge-backed runtime-consumption
  repair with one candidate row.
- `materializer_packet_member_zip_header_elide_v1`: bridge-backed
  runtime-consumption repair with one candidate row.

The bridge-less DFL1 parity work orders are still useful, but they do not claim
candidate custody they do not have. They emit same-runtime full-frame inflate
parity command hints and keep the correction budget gate closed. Bridge-backed
merge/header-elide rows additionally bind exact-readiness bridge reports,
candidate rows, materializer rerun hints, and exact-eval consumer rebuild hints.

## Authority Boundary

All new artifacts preserve:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The receiver repair queue is local work-order automation only. It is not score
authority, exact-eval dispatch authority, or permission to spend freed bytes on
targeted corrections before runtime consumption and component guards pass.

## Verification

- `ruff` passed on touched scheduler/tool/test files.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 16 passed.
- `git diff --check` passed.
- `tools/experiment_queue.py validate` passed for:
  - `.omx/research/frontier_operation_portfolio_explicit_receiver_repair_20260526T003426Z/dqs1_followup_queue.json`
  - `.omx/research/frontier_operation_portfolio_explicit_receiver_repair_20260526T003426Z/receiver_repair_queue.json`
- Executed all four receiver repair queue work-order commands successfully; the
  four current outputs wrote 5050, 4988, 12329, and 11593 bytes respectively.
- Re-ran the first explicit-input work-order command against the existing
  output; it returned `bytes_written=0` and
  `skipped_identical_existing_artifact=true`, proving queue retry idempotence.
- Captured the queue execution proof in
  `.omx/research/frontier_operation_portfolio_explicit_receiver_repair_20260526T003426Z/receiver_repair_queue_execution_smoke.json`.

## Next Engineering Move

Use the work-order outputs to compile actual repair execution rows: full-frame
DFL1 parity, packet-member merge runtime adapter repair, and submission runtime
manifest closure. Once one receiver proof clears, feed the proof-bound saved
bytes into component-guarded SegNet/PoseNet targeted correction acquisition.
