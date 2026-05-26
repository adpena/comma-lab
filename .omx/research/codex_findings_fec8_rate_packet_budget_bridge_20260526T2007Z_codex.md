# FEC8 Rate Packet Budget Bridge

UTC: 2026-05-26T20:07Z

## Verdict

Landed a scheduler bridge that turns an explicitly supplied receiver-closed
rate-packet materialization into a first-class repair/waterfill budget signal.
This is planning authority only. It does not claim score, promote, rank/kill, or
dispatch exact eval.

## What Changed

- Added `frontier_rate_attack_receiver_closed_rate_packet_materialization_signal.v1`.
- `build_receiver_closed_correction_budget(...)` now accepts:
  - `receiver_closed_rate_packet_paths`
  - `receiver_closed_rate_parent_paths`
- The refresh CLI now exposes:
  - `--receiver-closed-rate-packet`
  - `--receiver-closed-rate-parent`
- Rate-packet budget rows preserve candidate and parent archive/runtime custody,
  codec identity, selector wire bytes, entropy-position metadata, false-authority
  blockers, and saved-byte math.
- `build_frontier_rate_budget_preservation_plan(...)` now emits a direct
  operator-action row for receiver-closed rate-packet budget rows even when no
  legacy materializer backlog target matches the packet codec.

## Live FEC8 Anchor

Command:

```bash
.venv/bin/python tools/build_frontier_rate_attack_feedback_refresh.py \
  --action-summary none \
  --output-dir experiments/results/fec8_rate_packet_budget_bridge_20260526T2004Z_codex \
  --receiver-closed-rate-packet experiments/results/pr101_frame_exploit_selector_fec8_static_second_order_k16_20260526T1552Z_codex/packet_manifest.json \
  --receiver-closed-rate-parent experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packet_manifest.json \
  --repair-palette fec6-fixed-k16 \
  --candidate-limit 2
```

Output root:

`experiments/results/fec8_rate_packet_budget_bridge_20260526T2004Z_codex`

Receiver-closed budget:

- `receiver_closed_candidate_count`: 1
- `receiver_closed_saved_bytes_total`: 10
- candidate codec: `fec8_static_second_order_markov_k16`
- parent codec: `fec6_fixed_huffman_k16`
- archive delta vs parent: `-10` bytes
- selector payload wire delta: `-10` bytes
- `ready_for_budget_spend`: false

Targeted repair/waterfill:

- `targeted_component_correction_acquisition.active`: true
- `receiver_closed_saved_bytes_total`: 10
- `queue_actionable_acquisition_count`: 8
- `repair_dynamics_prior_active`: true
- `repair_dynamics_palette_probe_count`: 3

All authority fields remain false. The receiver side is still decode-only; the
repair and final rate attack remain encoder/compression-side planning and
materialization work.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py tools/build_frontier_rate_attack_feedback_refresh.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` -> 43 passed
- Live refresh CLI generated receiver budget, targeted correction acquisition,
  repair waterfill queue, receiver repair queue, operation materializer queue,
  operation chain queue, and autonomous chain queue.
- `tools/experiment_queue.py validate` passed for all six generated
  `experiment_queue.v1` artifacts in the live output root.

## Remaining Blockers

- The 10 bytes are available for local planning and waterfill allocation only.
- Component response must be measured before any repair budget spend.
- Exact auth eval remains required before any score/promotion claim.
- MLX/local rows can guide acquisition, but not contest authority.
