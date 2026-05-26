# FEC8 Rate Packet Budget Bridge V2 Custody Proof

UTC: 2026-05-26T20:09Z
Agent: codex
Lane: `lane_fec8_rate_packet_budget_bridge_20260526`

## Verdict

The receiver-closed rate-packet bridge is now stricter and less lossy. A rate
packet only becomes an active correction-budget signal when both candidate and
parent packet manifests are schema-correct, false-authority-clean, backed by
local archive files whose byte counts and SHA-256 hashes match the manifests,
and backed by existing runtime directories.

The FEC8 static second-order selector packet remains a rate-only planning
signal, not score/promotion/dispatch authority. Exact auth eval for this packet
already showed `[contest-CPU]` and `[contest-CUDA]` regressions, so the valid
use is preserving the 10-byte entropy-position win and feeding it into
component-guarded repair/waterfill planning.

## Live V2 Anchor

Command:

```bash
.venv/bin/python tools/build_frontier_rate_attack_feedback_refresh.py \
  --queue-id frontier_rate_attack_fec8_rate_packet_bridge_20260526_codex_v2 \
  --action-summary none \
  --skip-raw-retention-plan \
  --skip-mlx-retention-plan \
  --receiver-closed-rate-packet experiments/results/pr101_frame_exploit_selector_fec8_static_second_order_k16_clean_20260526_codex/packet_manifest.json \
  --receiver-closed-rate-parent experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packet_manifest.json \
  --output-dir .omx/research/frontier_rate_attack_feedback_refresh_fec8_rate_packet_bridge_20260526_codex_v2
```

Output root is intentionally ignored by `.gitignore`:

` .omx/research/frontier_rate_attack_feedback_refresh_fec8_rate_packet_bridge_20260526_codex_v2/`

Key artifacts:

- `receiver_closed_correction_budget.json`
- `rate_budget_preservation_plan.json`
- `targeted_component_correction_acquisition.json`
- `targeted_component_correction_queue.json`

Receiver-closed packet row:

- candidate codec: `fec8_static_second_order_markov_k16`
- parent codec: `fec6_fixed_huffman_k16`
- `saved_bytes_at_risk`: `10`
- `archive_byte_delta_vs_parent`: `-10`
- `archive_file_sha256_verified`: `true`
- `archive_file_bytes_verified`: `true`
- `source_archive_file_sha256_verified`: `true`
- `source_archive_file_bytes_verified`: `true`
- `submission_dir_verified`: `true`
- `source_submission_dir_verified`: `true`
- `ready_for_budget_spend`: `false`
- `score_claim`: `false`
- `promotion_eligible`: `false`

Targeted correction rows now carry the rate-packet signal without requiring
consumers to infer it from generic budget fields:

- `receiver_closed_saved_bytes`
- `saved_bytes_budget`
- `rate_packet_manifest_path`
- `parent_rate_packet_manifest_path`
- `candidate_compact_selector_codec`
- `parent_compact_selector_codec`
- `entropy_position`

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py tools/build_frontier_rate_attack_feedback_refresh.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` -> `44 passed`
- `.venv/bin/python tools/lane_maturity.py validate` -> `1421 lane(s) validated cleanly`

## Guard Added

`test_receiver_closed_rate_packet_manifest_refuses_unverified_archive_file`
corrupts the candidate archive after manifest creation and verifies the bridge
refuses receiver-closed status with byte and SHA mismatch blockers. This
prevents future packet budget rows from becoming planner signal from unverified
manifest claims.

## Remaining Blockers

- The bridge is L2, not L3. Contest CPU/CUDA gates remain false for this lane
  because the relevant exact auth result is negative.
- `ready_for_budget_spend=false` until component response is measured on the
  same candidate/reference context.
- `score_claim=false`, `promotion_eligible=false`, and
  `ready_for_exact_eval_dispatch=false` remain mandatory for the generated
  feedback artifacts.
