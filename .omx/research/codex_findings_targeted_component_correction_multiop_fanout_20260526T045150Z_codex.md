# Targeted Component Correction Multi-Op Fanout

Generated: 2026-05-26T04:51:50Z

## Verdict

Receiver-closed rate budget is now consumed by the local component-correction
queue as a bounded candidate-family matrix instead of collapsing each
receiver-closed candidate to one correction family. This moves the rate win
toward autonomous SegNet/PoseNet repair acquisition while preserving false
authority: no score claim, promotion, rank/kill, exact-eval dispatch, or budget
spend authority is granted by the queue.

## Code Change

- `src/comma_lab/scheduler/frontier_rate_attack_feedback.py` now selects
  targeted component-correction rows with
  `bounded_candidate_family_round_robin`.
- The targeted correction queue and each experiment metadata row carry
  `frontier_rate_attack_targeted_component_queue_selection_policy.v1`.
- `tools/build_frontier_rate_attack_feedback_refresh.py` propagates the refresh
  `candidate_limit` into the targeted component-correction queue writer, so the
  operator cap applies to this queue too.

## Live Artifact

Refresh output:

- `.omx/research/frontier_rate_attack_feedback_refresh_20260526T_multiop_component_fanout_v3/`

Key live results:

- Receiver-closed rate budget: 2 candidates, 414 saved bytes.
- Targeted component-correction acquisition: 10 actionable rows.
- Targeted component-correction queue: 6 experiments.
- Queue selection policy: `bounded_candidate_family_round_robin`.
- Selected receiver-closed candidates:
  - `packet_member_merge_cbe7d79124ba` with 258 saved bytes.
  - `packet_member_zip_header_elide_544c5f580ec2` with 156 saved bytes.
- Selected correction families:
  - `segnet_posenet_waterfill_region_repair`
  - `drop_within_selected_set_masked_boundary`
  - `inverse_scorer_cell_basis_expansion`

All six targeted-correction experiments still require local CPU component
advisory plus MLX response steps and keep `budget_spend_ready=false`.

## Eureka JSON Readout

The bounded refresh discovered 48 local-CPU eureka signals:

- 26 `decoder_q_pairset_drop_one`
- 22 `decoder_q_pairset_drop_two`
- 48 near-frontier observe-only rows

Best conservative gap in the scanned set is
`pairset_drop_one_rank024_pair0112` at about `2.5e-6` above the auth frontier.
Best drop-two rows are `pairset_drop_two_r029_017_p0259_0242` and
`pairset_drop_two_r029_021_p0259_0371` at about `2.834e-6` conservative gap.

The planner hint remains the important result: the drop-two cluster is close
enough to justify moving beyond drop-two into `learned_multi_drop`,
`drop_many_beam_pairwise_interaction_waterfill`,
`within_selected_set_mask_feather_probe`,
`master_gradient_constrained_low_sensitivity_drop`, and
`inverse_scorer_null_direction_masked_variant`.

## Verification

- `ruff` passed on touched scheduler, tool, and test files.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 22 passed.
- Live queue validation passed for:
  - targeted component-correction queue
  - DQS1 follow-up queue
  - receiver repair queue

## Remaining Gap

This is still local acquisition, not budget spend. The next highest-EV bridge is
to make the component-correction experiments harvest their local CPU/MLX
component responses back into the same Lagrangian accept/reject model, then
materialize only corrections whose measured
`delta_segnet + delta_posenet + lambda * delta_bytes` is negative under the
same-axis guard.
