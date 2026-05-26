# Codex Findings - FEC8 Targeted Component Queue Binding Repair

Generated: 2026-05-26T20:53Z
Author: Codex

## Scope

This pass resumed the receiver-closed FEC8 rate-packet bridge after the
component-correction queue failed on a synthesized candidate archive path. The
goal was to preserve signal, repair custody binding, rerun the local queue, and
advance the first executable receiver-consumed materializer proof without
claiming score authority.

## Bug Class Extincted

The targeted component acquisition and queue path was treating
`submission_dir/archive.zip` as if it were the candidate archive. For the FEC8
rate-packet archive, the real byte-closed candidate is:

`experiments/results/pr101_frame_exploit_selector_fec8_static_second_order_k16_clean_20260526_codex/archive.zip`

The missing synthesized path caused the original queue state to fail before the
local component response could run. The failed default state was preserved as
evidence:

`.omx/state/experiment_queue_frontier_rate_attack_fec8_rate_packet_bridge_20260526_codex_v2_component_correction.sqlite`

The code now carries explicit candidate archive and inflate paths from the
budget row into targeted component acquisition and refuses stale actionable rows
that omit those paths. The repaired state was run separately with an explicit
noncanonical-state rationale:

`.omx/state/experiment_queue_frontier_rate_attack_fec8_rate_packet_bridge_20260526_codex_v2_component_correction_repaired.sqlite`

Worker summary:

`.omx/research/frontier_rate_attack_feedback_refresh_fec8_rate_packet_bridge_20260526_codex_v2/targeted_component_correction_queue_worker_summary_repaired.json`

Result: 15/15 local queue steps succeeded, 0 failed.

## Local CPU and MLX Paired Evidence

Reference archive:

`experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`

Candidate archive:

`experiments/results/pr101_frame_exploit_selector_fec8_static_second_order_k16_clean_20260526_codex/archive.zip`

Reference `[macOS-CPU advisory]`: 0.19206131688110561, 178517 bytes.

Candidate `[macOS-CPU advisory]`: 0.1920546582915744, 178507 bytes.

Paired CPU delta: -0.000006658589531221714, entirely from 10 receiver-closed
bytes saved. SegNet and PoseNet deltas were 0.

Reference `[macOS-MLX research-signal]`: 0.19242568815842542, 178517 bytes.

Candidate `[macOS-MLX research-signal]`: 0.19241902956889423, 178507 bytes.

Paired MLX delta: -0.000006658589531221714.

Absolute candidate MLX-vs-CPU drift: 0.000364371277319836, dominated by SegNet
drift of 0.0003634184229007534. Paired candidate-vs-reference delta drift was
0.0, so MLX remains useful for this receiver-closed rate attack acquisition
class while still carrying no score, promotion, rank, or kill authority.

## Response Harvest

Refreshed aggregate:

`.omx/research/frontier_rate_attack_feedback_refresh_fec8_rate_packet_bridge_20260526_codex_v2/targeted_component_correction_response_harvest.json`

Rows: 4. All four local correction families became negative-Lagrangian local
acquisition candidates:

- `segnet_posenet_waterfill_region_repair`
- `drop_within_selected_set_masked_boundary`
- `inverse_scorer_cell_basis_expansion`
- `pose_stable_pair_frame_motion_correction`

`ready_for_budget_spend_count` remained 0. This is correct: the rate credit is
real locally, but spending that credit on distortion still requires a
receiver-consumed correction materializer, full-frame inflate parity, exact-axis
component response, and exact auth eval before any score or promotion claim.

Materialization request:

`.omx/research/frontier_rate_attack_feedback_refresh_fec8_rate_packet_bridge_20260526_codex_v2/targeted_component_correction_materialization_requests.json`

Request id:

`targeted_component_materialization_receiver_closed_rate_packet_lane_pr101_frame_exploit_selector_fec8_static_se_4ab4c1053240f68d_001`

Best measured local Lagrangian delta: -0.000006658589531221714.

Accepted-family sum: -0.000026634358124886855.

The materialization request stays fail-closed with blockers including:

- `receiver_consumed_correction_materializer_missing`
- `full_frame_inflate_parity_required_before_budget_spend`
- `exact_axis_component_response_required_before_budget_spend`
- `exact_auth_eval_required_before_score_or_promotion_claim`

## Receiver-Consumed Materializer Probe

The chain handoff exposed one locally executable materializer row:

`packet_member_zip_header_elide_v1`

Command output manifest:

`experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/frontier_targeted_component_correction_chain_materializers/targeted_component_chain_materializers/targeted_component_chain_targeted_component_materialization_receiver_closed_rate_packet_lane_pr101_frame_exploit_selector_fec8_static_se_4ab4c1053240f68d_001/packet_member_zip_header_elide_v1/manifest.json`

Runtime-consumption proof:

`experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/frontier_targeted_component_correction_chain_materializers/targeted_component_chain_materializers/targeted_component_chain_targeted_component_materialization_receiver_closed_rate_packet_lane_pr101_frame_exploit_selector_fec8_static_se_4ab4c1053240f68d_001/packet_member_zip_header_elide_v1/runtime_consumption_proof.json`

Result: byte-closed candidate emitted, receiver contract satisfied, but
`saved_bytes=0` and candidate SHA remained
`b44da5d54d34ce094c8fca7a37172b9ea6546d8a56eb960afa86252f56d11844`.
The materializer correctly refused promotion with `candidate_not_rate_positive`.
This makes `packet_member_zip_header_elide_v1` a local negative for this exact
FEC8 archive, not for other archive families.

## Retention and Storage

Retention plan:

`.omx/research/codex_fec8_targeted_component_retention_plan_20260526T2044Z.json`

Execution journal:

`.omx/research/codex_fec8_targeted_component_retention_plan_20260526T2044Z.json.journal.jsonl`

The plan moved two certified-rebuildable local CPU inflated raw directories to
`/Volumes/VertigoDataTier/pact_cold_store` with copy-verify-delete semantics:

- executed rows: 2
- local bytes reclaimed: 7324819200
- cold-store tier used: VertigoDataTier
- APDataStore stayed unused because the first tier had enough reserve

The advisory JSON, scorer-input cache hashes, MLX response JSON, provenance,
archive copies, and inflated-output manifests remain local. Only rebuildable
raw inflation payloads were moved.

## Next Actions

1. Build the receiver-consumed correction materializer for the four accepted
   local correction families, not just the packet header/materializer scaffold.
2. Add full-frame inflate parity and exact-axis component-response gates to the
   materializer handoff so a budget-spend candidate can become exact-eval ready.
3. Promote the paired MLX-vs-CPU drift summary into the acquisition model:
   absolute MLX offsets are real, but paired deltas are exact for this
   receiver-closed rate class.
4. Keep running storage retention through the tiered cold-store policy before
   broad component sweeps so raw inflation outputs do not again dominate local
   disk.
5. Use this result as the rate-credit substrate for Cascade C only after a
   receiver-consumed correction payload exists; local acquisition signal alone
   is not budget-spend authority.
